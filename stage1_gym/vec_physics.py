"""Vectorized Isaac Gym PhysX backend for many Go2-X5 actors in one sim."""

from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .config import LEG_NAMES, ARM_NAMES, FOOT_NAMES, LEG_HOME, ARM_KP, ARM_KD


class VecRobotState:
    def __init__(self, root, leg_pos, leg_vel, arm_pos, arm_vel, foot_pos, foot_vel, foot_force, collision_forces):
        self.root = root
        self.leg_pos = leg_pos
        self.leg_vel = leg_vel
        self.arm_pos = arm_pos
        self.arm_vel = arm_vel
        self.foot_pos = foot_pos
        self.foot_vel = foot_vel
        self.foot_force = foot_force
        self.collision_forces = collision_forces

    def finite(self):
        return all(np.isfinite(value).all() for value in vars(self).values())


class VecPhysics:
    def __init__(self, cfg, sim_device, num_envs, rng):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self.num_envs = num_envs
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.closed = False
        if sim_device == "cpu":
            device_id, gpu = 0, False
        elif sim_device.startswith("cuda:") and sim_device[5:].isdigit():
            device_id, gpu = int(sim_device[5:]), True
            if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {sim_device} is unavailable")
        else:
            raise ValueError("sim_device must be 'cpu' or 'cuda:N'")
        params = gymapi.SimParams()
        params.dt = cfg.physics_dt
        params.substeps = 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.use_gpu_pipeline = gpu
        params.physx.use_gpu = gpu
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 4
        params.physx.num_velocity_iterations = 1
        params.physx.num_threads = 4
        params.physx.contact_offset = 0.01
        params.physx.rest_offset = 0.0
        params.physx.bounce_threshold_velocity = 0.5
        params.physx.max_depenetration_velocity = 1.0
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        try:
            self.sim = self.gym.create_sim(device_id, -1, gymapi.SIM_PHYSX, params)
            if self.sim is None:
                raise RuntimeError("Isaac Gym could not create the PhysX simulation")
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0, 0, 1)
            plane.static_friction = 1.0
            plane.dynamic_friction = 1.0
            plane.restitution = 0.0
            self.gym.add_ground(self.sim, plane)
            self._create_robots(rng)
            self.gym.prepare_sim(self.sim)
            self.root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(num_envs, 13)
            self.dof = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(num_envs, self.num_dofs, 2)
            self.body = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(
                num_envs, self.num_bodies, 13)
            self.contact = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(
                num_envs, self.num_bodies, 3)
            self.device = self.dof.device
            self.actor_indices = torch.as_tensor(self.actor_indices, dtype=torch.int32, device=self.device)
            self.efforts = torch.zeros(num_envs, self.num_dofs, dtype=torch.float32, device=self.device)
            self.position_targets = torch.zeros_like(self.efforts)
            self.leg_indices = torch.as_tensor(self.legs, dtype=torch.long, device=self.device)
            self.torque_limits = torch.as_tensor(self.effort_limits, device=self.device)
            self.motor_strength_tensor = torch.ones(num_envs, 12, dtype=torch.float32, device=self.device)
            self.leg_offset_tensor = torch.zeros(num_envs, 12, dtype=torch.float32, device=self.device)
            self.motor_offset = np.zeros((num_envs, self.num_dofs), dtype=np.float32)
            self._refresh()
        except Exception:
            self.close()
            raise

    def _create_robots(self, rng):
        assets = Path(__file__).resolve().parent / "assets"
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        options.thickness = 0.01
        self.asset = self.gym.load_asset(self.sim, str(assets), "urdf/go2_x5.urdf", options)
        if self.asset is None:
            raise RuntimeError(f"Unable to load packaged robot asset in {assets}")
        names = list(self.gym.get_asset_dof_names(self.asset))
        bodies = list(self.gym.get_asset_rigid_body_names(self.asset))
        self.num_dofs = len(names)
        self.num_bodies = len(bodies)
        self.legs = np.array([names.index(n) for n in LEG_NAMES])
        self.arms = np.array([names.index(n) for n in ARM_NAMES])
        self.feet = np.array([bodies.index(n) for n in FOOT_NAMES])
        self.collision_bodies = np.array([
            i for i, n in enumerate(bodies) if any(part in n for part in ("base", "thigh", "calf", "Head", "x5_link"))
        ], dtype=int)
        self.home = np.zeros(self.num_dofs, dtype=np.float32)
        self.home[self.legs] = LEG_HOME
        self.props = self.gym.get_asset_dof_properties(self.asset)
        self.lower = self.props["lower"].copy()
        self.upper = self.props["upper"].copy()
        self.effort_limits = self.props["effort"][self.legs].copy()
        self.home = np.clip(self.home, self.lower, self.upper)
        self.props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.props["stiffness"].fill(1000.0)
        self.props["damping"].fill(100.0)
        self.props["driveMode"][self.legs] = gymapi.DOF_MODE_EFFORT
        self.props["stiffness"][self.legs] = 0.0
        self.props["damping"][self.legs] = 0.0
        self.props["stiffness"][self.arms] = ARM_KP
        self.props["damping"][self.arms] = ARM_KD
        grid = int(np.ceil(np.sqrt(self.num_envs)))
        spacing = 3.0
        self.envs, self.actors, self.actor_indices = [], [], []
        lower, upper = gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing)
        pose = gymapi.Transform()
        pose.p.z = self.cfg.reset_height
        for index in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, grid)
            actor = self.gym.create_actor(env, self.asset, pose, f"go2_x5_{index}", index, 1, 0)
            self.gym.set_actor_dof_properties(env, actor, self.props)
            self.envs.append(env)
            self.actors.append(actor)
            self.actor_indices.append(self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))

    def reset(self, env_ids, roots, positions, rng):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        c = self.cfg
        self.motor_offset[env_ids] = 0.0
        motor_strength = np.ones((len(env_ids), 12), dtype=np.float32)
        for local, env_id in enumerate(env_ids):
            props = self.props.copy()
            if c.randomize_dynamics:
                motor_strength[local] = rng.uniform(*c.motor_strength_range, size=12)
                self.motor_offset[env_id] = rng.uniform(-c.motor_offset_noise, c.motor_offset_noise, self.num_dofs)
                props["stiffness"][self.arms] = np.asarray(ARM_KP) * rng.uniform(0.5, 1.5, 6)
                props["damping"][self.arms] = np.asarray(ARM_KD) * rng.uniform(0.2, 2.0, 6)
            self.gym.set_actor_dof_properties(self.envs[env_id], self.actors[env_id], props)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.motor_strength_tensor[ids] = torch.as_tensor(motor_strength, device=self.device)
        self.leg_offset_tensor[ids] = torch.as_tensor(self.motor_offset[env_ids][:, self.legs], device=self.device)
        self.dof[ids, :, 0] = torch.as_tensor(positions, device=self.device)
        self.dof[ids, :, 1] = 0.0
        self.position_targets[ids] = torch.as_tensor(positions, device=self.device)
        self.efforts[ids] = 0.0
        self.root[ids] = torch.as_tensor(roots, device=self.device)
        actor_ids = self.actor_indices[ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof.reshape(-1, 2)),
                                             gymtorch.unwrap_tensor(actor_ids), len(env_ids))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root),
                                                     gymtorch.unwrap_tensor(actor_ids), len(env_ids))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.position_targets.reshape(-1)))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.efforts.reshape(-1)))
        state = self.state(env_ids)
        state.foot_force.fill(0)
        state.collision_forces.fill(0)
        return state

    def step(self, leg_target, arm_target):
        targets = np.repeat(self.home[None, :], self.num_envs, axis=0)
        targets[:, self.arms] = np.clip(arm_target + self.motor_offset[:, self.arms],
                                        self.lower[self.arms], self.upper[self.arms])
        self.position_targets[:] = torch.as_tensor(targets, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.position_targets.reshape(-1)))
        target_tensor = torch.as_tensor(leg_target, device=self.device)
        for _ in range(self.cfg.decimation):
            torques = (self.cfg.leg_kp * (target_tensor - self.dof[:, self.leg_indices, 0] + self.leg_offset_tensor)
                       - self.cfg.leg_kd * self.dof[:, self.leg_indices, 1]) * self.motor_strength_tensor
            torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
            self.efforts.zero_()
            self.efforts[:, self.leg_indices] = torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.efforts.reshape(-1)))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self._refresh()
        return self.state(), torques.cpu().numpy().copy()

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def state(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        root = self.root[env_ids].cpu().numpy().copy()
        dof = self.dof[env_ids].cpu().numpy()
        body = self.body[env_ids].cpu().numpy()
        contact = self.contact[env_ids].cpu().numpy()
        return VecRobotState(root, dof[:, self.legs, 0].copy(), dof[:, self.legs, 1].copy(),
                             dof[:, self.arms, 0].copy(), dof[:, self.arms, 1].copy(),
                             body[:, self.feet, :3].copy(), body[:, self.feet, 7:10].copy(),
                             contact[:, self.feet].copy(), contact[:, self.collision_bodies].copy())

    def close(self):
        if self.closed:
            return
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        self.closed = True
