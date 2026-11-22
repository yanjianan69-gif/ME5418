import ast
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_all_robot_meshes_are_packaged(self):
        urdf = ROOT / 'stage1_gym/assets/urdf/go2_x5.urdf'
        for mesh in ET.parse(urdf).findall('.//mesh'):
            path = (urdf.parent / mesh.attrib['filename']).resolve()
            self.assertTrue(path.is_file(), str(path))
            self.assertIn(ROOT, path.parents)
        for dae in (ROOT / 'stage1_gym/assets').rglob('*.dae'):
            for image in ET.parse(dae).findall('.//{*}image/{*}init_from'):
                self.assertTrue((dae.parent / image.text).is_file(), image.text)

    def test_self_contained_imports(self):
        forbidden = {'go1_gym', 'go1_gym_learn', 'modules', 'wandb', 'pytorch3d', 'cv2'}
        for source in list((ROOT / 'stage1_gym').glob('*.py')) + list((ROOT / 'stage1_rl').glob('*.py')):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                self.assertFalse(forbidden.intersection(name.split('.')[0] for name in names), str(source))
            self.assertNotIn('sys.path', source.read_text())


if __name__ == '__main__':
    unittest.main()
