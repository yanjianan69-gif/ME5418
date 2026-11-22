"""Isaac Gym PhysX simulation and mixed leg/arm joint control."""

from pathlib import Path

# Isaac Gym must precede torch (including indirect imports from RL libraries).
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .config import LEG_NAMES, ARM_NAMES, FOOT_NAMES, LEG_HOME, ARM_KP, ARM_KD
from .task import RobotState


class Physics:
    def __init__(self, cfg, sim_device, render_mode, rng):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.camera = None
        self.closed = False
        self.render_mode = render_mode
        if sim_device == "cpu":
            device_id, gpu = 0, False
        elif sim_device.startswith("cuda:") and sim_device[5:].isdigit():
            device_id, gpu = int(sim_device[5:]), True
            if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {sim_device} is unavailable; use sim_device='cpu'.")
        else:
            raise ValueError("sim_device must be 'cpu' or 'cuda:N'")
        params = gymapi.SimParams()
        params.dt = cfg.physics_dt
        params.substeps = 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        # GPU execution uses BOTH GPU PhysX and the GPU tensor pipeline.
        # Only the small standard-Gym observation/reward arrays cross to CPU.
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
            graphics_device = device_id if render_mode in ("human", "rgb_array") else -1
            self.sim = self.gym.create_sim(device_id, graphics_device,
                                            gymapi.SIM_PHYSX, params)
            if self.sim is None:
                raise RuntimeError("Isaac Gym could not create the PhysX simulation.")
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0, 0, 1)
            plane.static_friction = 1.0
            plane.dynamic_friction = 1.0
            plane.restitution = 0.0
            self.gym.add_ground(self.sim, plane)
            self._create_robot(rng)
            self.gym.prepare_sim(self.sim)
            self.root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
            self.dof = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            self.body = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
            self.contact = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
            self.device = self.dof.device
            self.actor_indices = torch.tensor([0], dtype=torch.int32, device=self.device)
            self.efforts = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
            self.position_targets = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
            self.leg_indices = torch.as_tensor(self.legs, dtype=torch.long, device=self.device)
            self.torque_limits = torch.as_tensor(self.effort_limits, device=self.device)
            self._refresh()
            if render_mode == "human":
                self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if self.viewer is None:
                    raise RuntimeError("Viewer creation failed; use render_mode=None for headless execution.")
            elif render_mode == "rgb_array":
                props = gymapi.CameraProperties()
                props.width = 1280
                props.height = 720
                self.camera = self.gym.create_camera_sensor(self.env, props)
                if self.camera < 0:
                    raise RuntimeError("Camera creation failed.")
        except Exception:
            self.close()
            raise

    def _create_robot(self, rng):
        assets = Path(__file__).resolve().parent / "assets"
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        options.flip_visual_attachments = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        options.angular_damping = 0.0
        options.linear_damping = 0.0
        options.armature = 0.0
        options.thickness = 0.01
        self.asset = self.gym.load_asset(self.sim, str(assets), "urdf/go2_x5.urdf", options)
        if self.asset is None:
            raise RuntimeError(f"Unable to load packaged robot asset in {assets}")
        names = list(self.gym.get_asset_dof_names(self.asset))
        bodies = list(self.gym.get_asset_rigid_body_names(self.asset))
        self.num_dofs = len(names)
        # URDF import order is not an action-space contract: explicitly map names.
        self.legs = np.array([names.index(n) for n in LEG_NAMES])
        self.arms = np.array([names.index(n) for n in ARM_NAMES])
        self.feet = np.array([bodies.index(n) for n in FOOT_NAMES])
        self.collision_bodies = np.array([
            i for i, n in enumerate(bodies)
            if any(part in n for part in ("base", "thigh", "calf", "Head", "x5_link"))
        ], dtype=int)
        self.home = np.zeros(self.num_dofs, dtype=np.float32)
        self.home[self.legs] = LEG_HOME
        self.props = self.gym.get_asset_dof_properties(self.asset)
        self.lower = self.props["lower"].copy()
        self.upper = self.props["upper"].copy()
        self.effort_limits = self.props["effort"][self.legs].copy()
        self.home = np.clip(self.home, self.lower, self.upper)
        self.props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.props["stiffness"].fill(1000.0)  # retain a closed gripper
        self.props["damping"].fill(100.0)
        self.props["driveMode"][self.legs] = gymapi.DOF_MODE_EFFORT
        self.props["stiffness"][self.legs] = 0.0
        self.props["damping"][self.legs] = 0.0
        self.props["stiffness"][self.arms] = ARM_KP
        self.props["damping"][self.arms] = ARM_KD
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        pose = gymapi.Transform()
        pose.p.z = self.cfg.reset_height
        self.actor = self.gym.create_actor(self.env, self.asset, pose, "go2_x5", 0, 1, 0)
        self.gym.set_actor_dof_properties(self.env, self.actor, self.props)
        self.instance_randomization = {"friction": 1.0, "arm_mass_scales": {}}
        if self.cfg.randomize_dynamics:
            friction = float(rng.uniform(*self.cfg.friction_range))
            shapes = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
            for shape in shapes:
                shape.friction = friction
                shape.restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(self.env, self.actor, shapes)
            body_props = self.gym.get_actor_rigid_body_properties(self.env, self.actor)
            for i, name in enumerate(bodies):
                if name.startswith("x5_"):
                    scale = float(rng.uniform(*self.cfg.arm_mass_scale))
                    offset = rng.uniform(-self.cfg.arm_com_noise, self.cfg.arm_com_noise, 3)
                    body_props[i].mass *= scale
                    com = body_props[i].com
                    body_props[i].com = gymapi.Vec3(com.x + offset[0], com.y + offset[1], com.z + offset[2])
                    self.instance_randomization["arm_mass_scales"][name] = scale
            self.gym.set_actor_rigid_body_properties(self.env, self.actor, body_props, recomputeInertia=True)
            self.instance_randomization["friction"] = friction

    def reset(self, root, positions, rng):
        c = self.cfg
        self.motor_strength = np.ones(12, dtype=np.float32)
        self.motor_offset = np.zeros(self.num_dofs, dtype=np.float32)
        props = self.props.copy()
        if c.randomize_dynamics:
            self.motor_strength[:] = rng.uniform(*c.motor_strength_range, size=12)
            self.motor_offset[:] = rng.uniform(-c.motor_offset_noise, c.motor_offset_noise, self.num_dofs)
            props["stiffness"][self.arms] = np.asarray(ARM_KP) * rng.uniform(0.5, 1.5, 6)
            props["damping"][self.arms] = np.asarray(ARM_KD) * rng.uniform(0.2, 2.0, 6)
        # Properties -> DOFs -> root. No articulation-property/DOF writes after root reset.
        self.gym.set_actor_dof_properties(self.env, self.actor, props)
        self.motor_strength_tensor = torch.as_tensor(self.motor_strength, device=self.device)
        self.leg_offset_tensor = torch.as_tensor(self.motor_offset[self.legs], device=self.device)
        self.dof[:, 0] = torch.as_tensor(positions, device=self.device)
        self.dof[:, 1] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof),
                                             gymtorch.unwrap_tensor(self.actor_indices), 1)
        self.position_targets[:] = torch.as_tensor(positions, device=self.device)
        self.efforts.zero_()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.position_targets))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.efforts))
        self.root[0] = torch.as_tensor(root, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root),
                                                    gymtorch.unwrap_tensor(self.actor_indices), 1)
        # Observations use the written root/DOFs directly. Do not simulate a
        # hidden step or return cached body/contact state from the old episode.
        state = self.state()
        state.foot_force.fill(0)
        state.collision_forces.fill(0)
        return state

    def step(self, leg_target, arm_target):
        targets = self.home.copy()
        targets[self.arms] = np.clip(arm_target + self.motor_offset[self.arms],
                                    self.lower[self.arms], self.upper[self.arms])
        self.position_targets[:] = torch.as_tensor(targets, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.position_targets))
        target_tensor = torch.as_tensor(leg_target, device=self.device)
        for _ in range(self.cfg.decimation):
            torques = (self.cfg.leg_kp * (target_tensor - self.dof[self.leg_indices, 0] + self.leg_offset_tensor)
                       - self.cfg.leg_kd * self.dof[self.leg_indices, 1]) * self.motor_strength_tensor
            torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
            self.efforts.zero_()
            self.efforts[self.leg_indices] = torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.efforts))
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

    def state(self):
        dof, body, contact = self.dof.cpu().numpy(), self.body.cpu().numpy(), self.contact.cpu().numpy()
        return RobotState(
            self.root[0].cpu().numpy().copy(), dof[self.legs, 0].copy(), dof[self.legs, 1].copy(),
            dof[self.arms, 0].copy(), dof[self.arms, 1].copy(), body[self.feet, :3].copy(),
            body[self.feet, 7:10].copy(), contact[self.feet].copy(), contact[self.collision_bodies].copy(),
        )

    def render(self):
        x, y, z = self.root[0, :3].tolist()
        eye = gymapi.Vec3(x + 2.0, y + 2.0, z + 1.3)
        target = gymapi.Vec3(x, y, z)
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                raise RuntimeError("Viewer closed. Call env.close() and stop stepping.")
            self.gym.viewer_camera_look_at(self.viewer, None, eye, target)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            return None
        if self.camera is not None:
            self.gym.set_camera_location(self.camera, self.env, eye, target)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            image = self.gym.get_camera_image(self.sim, self.env, self.camera, gymapi.IMAGE_COLOR)
            return image.reshape(720, 1280, 4)[:, :, :3].copy()
        return None

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        self.camera = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        self.closed = True
