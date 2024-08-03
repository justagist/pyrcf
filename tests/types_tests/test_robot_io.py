from typing import List
import pytest
import copy
import numpy as np
from pyrcf.core.types import EndEffectorStates, Pose3D, Twist, JointStates, RobotCmd


def check_ee_state_values_match(
    ee_state: EndEffectorStates,
    ee_names: List[str],
    ee_poses: List[Pose3D],
    ee_twists: List[Twist],
    ee_contact_states: List[int],
    ee_contact_forces: List[np.ndarray],
):
    assert ee_state.ee_names == ee_names
    assert np.all(ee_state.ee_poses == ee_poses)
    assert np.all(ee_state.ee_twists == ee_twists)
    assert np.all(ee_state.contact_states == ee_contact_states)
    assert all(
        np.all(ee_state.contact_forces[n] == ee_contact_forces[n])
        for n in range(len(ee_contact_forces))
    )


class TestEndEffectorStates:

    @pytest.fixture(scope="class")
    def ee_names(self):
        return ["ee_1", "ee_2", "ee_3"]

    @pytest.fixture(scope="class")
    def ee_poses(self):
        return [
            Pose3D(position=np.ones(3), orientation=np.array([0, 0, 0, 1])),
            Pose3D(position=np.array([1, 2, 3]), orientation=np.array([0, 0, 0, 1])),
            Pose3D(position=np.array([-1, -2, 3]), orientation=np.array([1, 0, 0, 0])),
        ]

    @pytest.fixture(scope="class")
    def ee_twists(self):
        return [
            Twist(linear=np.ones(3), angular=np.array([0, 0, 1])),
            Twist(linear=np.array([1.5, 2.3, 3]), angular=np.array([0.6, 2.5, 1])),
            Twist(linear=np.array([-1, -2, 3]), angular=np.array([1.0, -3.5, 0.7])),
        ]

    @pytest.fixture(scope="class")
    def ee_forces(self):
        return [[0.0, 1.0, -3.0], [1.0, -2.0, 0.02], [0.0, 0.0, 0.0]]

    @pytest.fixture(scope="class")
    def ee_contact_states(self):
        return [1, 0, 0]

    @pytest.fixture(scope="class")
    def ee_obj(self, ee_names, ee_poses, ee_twists, ee_contact_states, ee_forces):
        return EndEffectorStates(
            ee_names=ee_names,
            ee_poses=ee_poses,
            ee_twists=ee_twists,
            contact_states=ee_contact_states,
            contact_forces=ee_forces,
        )

    def test_get_method_success(
        self,
        ee_obj: EndEffectorStates,
        ee_names,
        ee_poses,
        ee_twists,
        ee_contact_states,
        ee_forces,
    ):
        def test_get_method(name):
            out1 = ee_obj[name]
            out2 = ee_obj.get_state_of(ee_name=name)
            assert out1[0] == out2[0]
            assert out1[1] == out2[1]
            assert out1[2] == out2[2]
            assert all(out1[3] == out2[3])

        def test_vals_from_get_method(n, ee_name, pose, twist, contact_state, force):
            assert ee_name == ee_names[n]
            assert pose == ee_poses[n]
            assert twist == ee_twists[n]
            assert contact_state == ee_contact_states[n]
            assert all(force == ee_forces[n])

        for n, ee_name in enumerate(ee_obj.ee_names):

            test_get_method(name=ee_name)

            pose, twist, contact_state, force = ee_obj[ee_name]

            test_vals_from_get_method(n, ee_name, pose, twist, contact_state, force)

    def test_get_by_invalid_ee_name(self, ee_obj: EndEffectorStates):
        with pytest.raises(KeyError, match=r".*not found in this object.*"):
            ee_obj.get_state_of(ee_name="invalid_name")

        with pytest.raises(KeyError, match=r".*not found in this object.*"):
            _ = ee_obj["invalid_name2"]

    def test_get_non_existent_vals(self, ee_obj: EndEffectorStates):
        ee_tst_obj = copy.deepcopy(ee_obj)
        ee_tst_obj.ee_names = None
        with pytest.raises(AttributeError, match=r".*`ee_names` attribute is not defined.*"):
            ee_tst_obj.get_state_of(ee_name="invalid_name")

        with pytest.raises(AttributeError, match=r".*`ee_names` attribute is not defined.*"):
            _ = ee_tst_obj["invalid_name2"]

    def test_extend_method(self, ee_obj: EndEffectorStates):
        ee_s1 = copy.deepcopy(ee_obj)
        ee_s2 = EndEffectorStates(
            ee_names=["ee_4", "ee_5", "ee_6"],
            ee_poses=[
                Pose3D(position=np.ones(3) * 5, orientation=np.array([0, 1, 0, 0])),
                Pose3D(position=np.array([1, 2, 3]) * 5, orientation=np.array([0, 0, 1, 0])),
                Pose3D(
                    position=np.array([-1, -2, 3]) * 5,
                    orientation=np.array([1, 0, 0, 0]),
                ),
            ],
            ee_twists=[
                Twist(linear=np.ones(3) * 5, angular=np.array([0, 0, 1]) * 5),
                Twist(
                    linear=np.array([1.5, 2.3, 3]) * 5,
                    angular=np.array([0.6, 2.5, 1]) * 5,
                ),
                Twist(
                    linear=np.array([-1, -2, 3]) * 5,
                    angular=np.array([1.0, -3.5, 0.7]) * 5,
                ),
            ],
            contact_states=None,
            contact_forces=None,
        )
        ee_s1.extend(ee_s2)
        check_ee_state_values_match(
            ee_s1,
            ["ee_1", "ee_2", "ee_3", "ee_4", "ee_5", "ee_6"],
            [
                Pose3D(
                    position=np.array([1.0000, 1.0000, 1.0000]),
                    orientation=np.array([0, 0, 0, 1]),
                ),
                Pose3D(position=np.array([1, 2, 3]), orientation=np.array([0, 0, 0, 1])),
                Pose3D(position=np.array([-1, -2, 3]), orientation=np.array([1, 0, 0, 0])),
                Pose3D(
                    position=np.array([5.0000, 5.0000, 5.0000]),
                    orientation=np.array([0, 1, 0, 0]),
                ),
                Pose3D(position=np.array([5, 10, 15]), orientation=np.array([0, 0, 1, 0])),
                Pose3D(position=np.array([-5, -10, 15]), orientation=np.array([1, 0, 0, 0])),
            ],
            [
                Twist(
                    linear=np.array([1.0000, 1.0000, 1.0000]),
                    angular=np.array([0, 0, 1]),
                ),
                Twist(
                    linear=np.array([1.5000, 2.3000, 3.0000]),
                    angular=np.array([0.6000, 2.5000, 1.0000]),
                ),
                Twist(
                    linear=np.array([-1, -2, 3]),
                    angular=np.array([1.0000, -3.5000, 0.7000]),
                ),
                Twist(
                    linear=np.array([5.0000, 5.0000, 5.0000]),
                    angular=np.array([0, 0, 5]),
                ),
                Twist(
                    linear=np.array([7.5000, 11.5000, 15.0000]),
                    angular=np.array([3.0000, 12.5000, 5.0000]),
                ),
                Twist(
                    linear=np.array([-5, -10, 15]),
                    angular=np.array([5.0000, -17.5000, 3.5000]),
                ),
            ],
            np.array([1, 0, 0, None, None, None]),
            [
                np.array([0.0000, 1.0000, -3.0000]),
                np.array([1.0000, -2.0000, 0.0200]),
                np.array([0.0000, 0.0000, 0.0000]),
                None,
                None,
                None,
            ],
        )

    def test_update_from_method(self):
        ee_s1 = EndEffectorStates(
            ee_names=["ee_1", "ee_2", "ee_3", "ee_4", "ee_5", "ee_6"],
            ee_poses=[
                Pose3D(
                    position=np.array([1.0000, 1.0000, 1.0000]),
                    orientation=np.array([0, 0, 0, 1]),
                ),
                Pose3D(position=np.array([1, 2, 3]), orientation=np.array([0, 0, 0, 1])),
                Pose3D(position=np.array([-1, -2, 3]), orientation=np.array([1, 0, 0, 0])),
                Pose3D(
                    position=np.array([5.0000, 5.0000, 5.0000]),
                    orientation=np.array([0, 1, 0, 0]),
                ),
                Pose3D(position=np.array([5, 10, 15]), orientation=np.array([0, 0, 1, 0])),
                Pose3D(position=np.array([-5, -10, 15]), orientation=np.array([1, 0, 0, 0])),
            ],
            ee_twists=[
                Twist(
                    linear=np.array([1.0000, 1.0000, 1.0000]),
                    angular=np.array([0, 0, 1]),
                ),
                Twist(
                    linear=np.array([1.5000, 2.3000, 3.0000]),
                    angular=np.array([0.6000, 2.5000, 1.0000]),
                ),
                Twist(
                    linear=np.array([-1, -2, 3]),
                    angular=np.array([1.0000, -3.5000, 0.7000]),
                ),
                Twist(
                    linear=np.array([5.0000, 5.0000, 5.0000]),
                    angular=np.array([0, 0, 5]),
                ),
                Twist(
                    linear=np.array([7.5000, 11.5000, 15.0000]),
                    angular=np.array([3.0000, 12.5000, 5.0000]),
                ),
                Twist(
                    linear=np.array([-5, -10, 15]),
                    angular=np.array([5.0000, -17.5000, 3.5000]),
                ),
            ],
            contact_states=np.array([1, 0, 0, None, None, None], dtype=object),
            contact_forces=[
                np.array([0.0000, 1.0000, -3.0000]),
                np.array([1.0000, -2.0000, 0.0200]),
                np.array([0.0000, 0.0000, 0.0000]),
                None,
                None,
                None,
            ],
        )
        ee_s2 = EndEffectorStates(
            ee_names=["ee_5", "ee_3", "ee_5"],
            ee_poses=[
                Pose3D(position=np.ones(3) * 7, orientation=np.array([0, 0, 0, 1])),
                Pose3D(position=np.array([1, 2, 3]) * 7, orientation=np.array([0, 0, 0, 1])),
                Pose3D(
                    position=np.array([-1, -2, 3]) * 7,
                    orientation=np.array([0, 0, 0, 1]),
                ),
            ],
            ee_twists=[
                Twist(linear=np.ones(3) * 7, angular=np.array([0, 0, 1]) * 7),
                Twist(
                    linear=np.array([1.5, 2.3, 3]) * 7,
                    angular=np.array([0.6, 2.5, 1]) * 7,
                ),
                Twist(
                    linear=np.array([-1, -2, 3]) * 7,
                    angular=np.array([1.0, -3.5, 0.7]) * 7,
                ),
            ],
            contact_states=[1, 1, 1],
            contact_forces=[np.zeros(3) for _ in range(3)],
        )

        ee_s1.update_from(ee_s2)
        check_ee_state_values_match(
            ee_s1,
            ["ee_1", "ee_2", "ee_3", "ee_4", "ee_5", "ee_6"],
            [
                Pose3D(
                    position=np.array([1.0000, 1.0000, 1.0000]),
                    orientation=np.array([0, 0, 0, 1]),
                ),
                Pose3D(position=np.array([1, 2, 3]), orientation=np.array([0, 0, 0, 1])),
                Pose3D(position=np.array([7, 14, 21]), orientation=np.array([0, 0, 0, 1])),
                Pose3D(
                    position=np.array([5.0000, 5.0000, 5.0000]),
                    orientation=np.array([0, 1, 0, 0]),
                ),
                Pose3D(position=np.array([-7, -14, 21]), orientation=np.array([0, 0, 0, 1])),
                Pose3D(position=np.array([-5, -10, 15]), orientation=np.array([1, 0, 0, 0])),
            ],
            [
                Twist(
                    linear=np.array([1.0000, 1.0000, 1.0000]),
                    angular=np.array([0, 0, 1]),
                ),
                Twist(
                    linear=np.array([1.5000, 2.3000, 3.0000]),
                    angular=np.array([0.6000, 2.5000, 1.0000]),
                ),
                Twist(
                    linear=np.array([10.5000, 16.1000, 21.0000]),
                    angular=np.array([4.2000, 17.5000, 7.0000]),
                ),
                Twist(
                    linear=np.array([5.0000, 5.0000, 5.0000]),
                    angular=np.array([0, 0, 5]),
                ),
                Twist(
                    linear=np.array([-7, -14, 21]),
                    angular=np.array([7.0000, -24.5000, 4.9000]),
                ),
                Twist(
                    linear=np.array([-5, -10, 15]),
                    angular=np.array([5.0000, -17.5000, 3.5000]),
                ),
            ],
            np.array([1, 0, 1, None, 1, None], dtype=object),
            [
                np.array([0.0000, 1.0000, -3.0000]),
                np.array([1.0000, -2.0000, 0.0200]),
                np.array([0.0000, 0.0000, 0.0000]),
                None,
                np.array([0.0000, 0.0000, 0.0000]),
                None,
            ],
        )


def check_joint_state_values_match(
    joint_states: JointStates,
    j_names: List[str],
    j_positions: np.ndarray,
    j_velocities: np.ndarray,
    j_efforts: np.ndarray,
):
    assert joint_states.joint_names == j_names
    assert np.all(joint_states.joint_positions == j_positions)
    assert np.all(joint_states.joint_velocities == j_velocities)
    assert np.all(joint_states.joint_efforts == j_efforts)


class TestJointStates:

    @pytest.fixture(scope="class")
    def joint_positions(self):
        return np.array([0.0, 0.02, 1.0, -100])

    @pytest.fixture(scope="class")
    def joint_velocities(self):
        return np.array([1, 2, 3, -12])

    @pytest.fixture(scope="class")
    def joint_efforts(self):
        return np.array([0.1, 0.2, -0.4, 0.56])

    @pytest.fixture(scope="class")
    def joint_names(self):
        return ["j1", "j2", "j3", "j4"]

    @pytest.fixture(scope="class")
    def js_obj(self, joint_positions, joint_velocities, joint_efforts, joint_names):
        return JointStates(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_efforts=joint_efforts,
            joint_names=joint_names,
        )

    def test_get_method_success(
        self,
        js_obj: JointStates,
        joint_positions,
        joint_velocities,
        joint_efforts,
        joint_names,
    ):
        def test_get_method(name):
            assert js_obj[name] == js_obj.get_state_of(joint_name=name)

        def test_vals_from_get_method(n, joint_name, pos, vel, torque):
            assert joint_name == joint_names[n]
            assert pos == joint_positions[n]
            assert vel == joint_velocities[n]
            assert torque == joint_efforts[n]

        for n, joint_name in enumerate(js_obj.joint_names):

            test_get_method(name=joint_name)

            pos, vel, torque = js_obj[joint_name]

            test_vals_from_get_method(n, joint_name, pos, vel, torque)

    def test_get_by_invalid_joint_name(self, js_obj: JointStates):
        with pytest.raises(KeyError, match=r".*not found in this object.*"):
            js_obj.get_state_of(joint_name="invalid_name")

        with pytest.raises(KeyError, match=r".*not found in this object.*"):
            _ = js_obj["invalid_name2"]

    def test_get_non_existent_vals(self, js_obj: JointStates):
        test_js_obj = copy.deepcopy(js_obj)
        test_js_obj.joint_names = None
        with pytest.raises(AttributeError, match=r".*`joint_names` attribute is not defined.*"):
            test_js_obj.get_state_of(joint_name="invalid_name")

        with pytest.raises(AttributeError, match=r".*`joint_names` attribute is not defined.*"):
            _ = test_js_obj["invalid_name2"]

    def test_extend_method(self, js_obj: JointStates):
        js1 = copy.deepcopy(js_obj)
        js2 = JointStates(
            joint_positions=np.array([0.033, 20.02, -11.0, 00]),
            joint_velocities=np.array([1.0, -2, 0.113, 1.2]),
            joint_efforts=np.array([0.101, 23.2, 10.4, -990.56]),
            joint_names=["j30", "j23", "j243", "j4544"],
        )
        js1.extend(js2)
        check_joint_state_values_match(
            js1,
            ["j1", "j2", "j3", "j4", "j30", "j23", "j243", "j4544"],
            np.array([0.0000, 0.0200, 1.0000, -100.0000, 0.0330, 20.0200, -11.0000, 0.0000]),
            np.array([1.0000, 2.0000, 3.0000, -12.0000, 1.0000, -2.0000, 0.1130, 1.2000]),
            np.array([0.1000, 0.2000, -0.4000, 0.5600, 0.1010, 23.2000, 10.4000, -990.5600]),
        )

    def test_update_method(self):
        js1 = JointStates(
            joint_positions=np.array(
                [0.0000, 0.0200, 1.0000, -100.0000, 0.0330, 20.0200, -11.0000, 0.0000]
            ),
            joint_velocities=np.array(
                [1.0000, 2.0000, 3.0000, -12.0000, 1.0000, -2.0000, 0.1130, 1.2000]
            ),
            joint_efforts=np.array(
                [0.1000, 0.2000, -0.4000, 0.5600, 0.1010, 23.2000, 10.4000, -990.5600]
            ),
            joint_names=["j1", "j2", "j3", "j4", "j30", "j23", "j243", "j4544"],
        )
        js2 = JointStates(
            joint_positions=np.ones(5),
            joint_velocities=np.array([1, 2, 3, 44, -12]),
            joint_efforts=np.array([1, 1, 55, -0.66, -990.56]),
            joint_names=["j23", "j3", "j30", "j2", "j3"],
        )
        js1.update_from(js2)
        check_joint_state_values_match(
            js1,
            ["j1", "j2", "j3", "j4", "j30", "j23", "j243", "j4544"],
            np.array([0.0000, 1.0000, 1.0000, -100.0000, 1.0000, 1.0000, -11.0000, 0.0000]),
            np.array([1.0000, 44.0000, -12.0000, -12.0000, 3.0000, 1.0000, 0.1130, 1.2000]),
            np.array([0.1000, -0.6600, -990.5600, 0.5600, 55, 1.0000, 10.4000, -990.5600]),
        )


class TestRobotCmd:

    def test_create_zeros_method(self):

        cmd: RobotCmd = RobotCmd.createZeros(7)

        assert len(cmd.Kp) == 7
        assert len(cmd.Kd) == 7
        assert len(cmd.joint_commands.joint_positions) == 7
        assert len(cmd.joint_commands.joint_velocities) == 7
        assert len(cmd.joint_commands.joint_efforts) == 7

        assert all(cmd.Kp == np.zeros(7))
        assert all(cmd.Kd == np.zeros(7))
        assert all(cmd.joint_commands.joint_positions == np.zeros(7))
        assert all(cmd.joint_commands.joint_velocities == np.zeros(7))
        assert all(cmd.joint_commands.joint_efforts == np.zeros(7))

        assert cmd.joint_commands.joint_names is None

    def test_create_zeros_with_names_method(self):
        cmd: RobotCmd = RobotCmd.createZeros(3, ["j1", "j2", "j3"])
        assert len(cmd.joint_commands.joint_names) == 3

    def test_create_zeros_with_wrong_names_lent(self):
        with pytest.raises(AssertionError):
            _: RobotCmd = RobotCmd.createZeros(7, ["j1", "j2", "j3"])

    def test_kp_kd_setter(self):
        cmd: RobotCmd = RobotCmd.createZeros(7)
        assert all(cmd.Kp == np.zeros(7))
        assert all(cmd.Kd == np.zeros(7))

        cmd.Kp = 3.0
        assert all(cmd.Kp == np.ones(7) * 3.0)

        cmd.Kd = 4.0
        assert all(cmd.Kd == np.ones(7) * 4.0)

    def test_kp_kd_setter_assertions(self):
        cmd: RobotCmd = RobotCmd.createZeros(7)

        with pytest.raises(AssertionError, match=r".*should match dimension of joint_positions.*"):
            cmd.Kp = [1, 2, 3]

        with pytest.raises(AssertionError, match=r".*should match dimension of joint_velocities.*"):
            cmd.Kd = [2, 3]

        with pytest.raises(AssertionError, match=r".*should be positive.*"):
            cmd.Kp = [2, -3, 0, 1, 2, 3, 7]

        with pytest.raises(AssertionError, match=r".*should be positive.*"):
            cmd.Kd = -0.2
