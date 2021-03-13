import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
import time
from data_util import pc_normalize


def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])


def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)


def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat)


def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)


def check_pairwise_collisions(bodies):
    for body1 in bodies:
        for body2 in bodies:
            if body1 != body2 and check_state_collision(body1, body2):
                return True
    return False


def check_state_collision(body1, body2):
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0., physicsClientId=0)) != 0


class Kuka:
    def __init__(self, urdfRootPath="models/", timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.0
        self.maxForce = 200.
        self.fingerAForce = 2.5
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # rest poses for null space
        self.rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]
        # joint damping coefficients
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]
        objects = p.loadSDF(self.urdfRootPath+"kuka_iiwa/kuka_with_gripper2.sdf")
        self.kukaUid = objects[0]
        self.numJoints = p.getNumJoints(self.kukaUid)
        self.endEffectorPos = [0.55, 0.0, 0.6]
        self.endEffectorAngle = np.pi
        self.fingerAngle = 0.0
        self.jointPositions = [0.0070825, 0.380528, - 0.009961, - 1.363558, 0.0037537, 1.397523, - 0.00280725,
                               np.pi, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0]
        self.motorNames = []
        self.motorIndices = []
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            jointInfo = p.getJointInfo(self.kukaUid, jointIndex)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(jointIndex)

    def reset(self):
        self.endEffectorPos = [0.55, 0.0, 0.6]
        self.endEffectorAngle = np.pi
        self.fingerAngle = 0.0
        self.jointPositions = [0.0070825, 0.380528, - 0.009961, - 1.363558, 0.0037537, 1.397523, - 0.00280725,
                               np.pi, 0.00000, 0.0, 0.0, 0.00000, 0.0, 0.0]
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
        finger_state = p.getJointState(self.kukaUid, 8)
        finger_angle = -finger_state[0]
        finger_force = finger_state[3]
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
        observation.extend([finger_angle, finger_force])
        return observation

    def applyAction(self, motorCommands):
        if self.useInverseKinematics:
            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            df = motorCommands[4]

            self.endEffectorPos[0] = min(max(self.endEffectorPos[0] + dx, 0.35), 0.75)
            self.endEffectorPos[1] = min(max(self.endEffectorPos[1] + dy, -0.2), 0.2)
            self.endEffectorPos[2] = min(max(self.endEffectorPos[2] + dz, 0.25), 0.65)
            self.fingerAngle = min(max(self.fingerAngle + df, 0.0), 0.4)
            self.endEffectorAngle += da
            pos = self.endEffectorPos
            orn = p.getQuaternionFromEuler([np.pi, 0, np.pi])
            self.setInverseKine(pos, orn, self.fingerAngle)
        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.kukaUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=motorCommands[action],
                                        force=self.maxForce)

    def setInverseKine(self, pos, orn, fingerAngle, useSimulation=1):
        useSimulation = useSimulation and self.useSimulation
        if self.useNullSpace == 1:
            if self.useOrientation == 1:
                jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos,
                                                          orn, self.ll, self.ul, self.jr, self.rp)
            else:
                jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                          self.kukaEndEffectorIndex,
                                                          pos,
                                                          lowerLimits=self.ll,
                                                          upperLimits=self.ul,
                                                          jointRanges=self.jr,
                                                          restPoses=self.rp)
        else:
            if self.useOrientation == 1:
                jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                          self.kukaEndEffectorIndex,
                                                          pos,
                                                          orn,
                                                          jointDamping=self.jd)
            else:
                jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)

        if useSimulation:
            for i in range(self.kukaEndEffectorIndex + 1):
                p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=self.maxForce,
                                        maxVelocity=self.maxVelocity,
                                        positionGain=0.3,
                                        velocityGain=1)
        else:
            # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range(self.kukaEndEffectorIndex + 1):
                p.resetJointState(self.kukaUid, i, jointPoses[i])
        # fingers
        p.setJointMotorControl2(self.kukaUid,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=self.endEffectorAngle,
                                force=self.maxForce)
        p.setJointMotorControl2(self.kukaUid,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=-fingerAngle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.kukaUid,
                                11,
                                p.POSITION_CONTROL,
                                targetPosition=fingerAngle,
                                force=self.fingerBForce)
        p.setJointMotorControl2(self.kukaUid,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)
        p.setJointMotorControl2(self.kukaUid,
                                13,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)


class KukaCamGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot="models/",
                 actionRepeat=3,
                 isEnableSelfCollision=True,
                 renders=False,
                 image_output=True,
                 mode='de',
                 width=128):
        print('******************************************\n'
              'Create Environment: \n'
              '     render?:', renders, ';\n'
              '     output image?:', image_output, ';\n'
              '     mode:', mode, ';\n'
              '     image_width:', width, '\n'
              '******************************************')
        self.seed()
        self._timeStep = 0.02
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._renders = renders
        self._image_output = image_output
        self._mode = mode
        self._width = width
        self._height = self._width
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.0, 230, -40, [0.55, 0, 0])
        else:
            p.connect(p.DIRECT)
        self._action_dim = 5
        self._action_bound = 1
        action_high = np.array([self._action_bound] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(6 if self._mode == 'de' else 4, self._height, self._width),
                                            dtype=np.uint8)
        self.viewer = None
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)
        p.loadURDF(self._urdfRoot+"floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF(self._urdfRoot+"table_collision/table.urdf", [0.6, 0, -0.625], useFixedBase=True)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._block = p.loadURDF(self._urdfRoot+"box_green.urdf")
        self._cup = p.loadURDF(self._urdfRoot+"cup/cup.urdf")
        self._range = 0.15

    def reset(self):
        collision = True
        while collision:
            set_pose(self._block, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                             np.random.uniform(-self._range, self._range), 0.025),
                                       [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            set_pose(self._cup, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                           np.random.uniform(-self._range, self._range), 0.0745),
                                     [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            collision = check_pairwise_collisions([self._block, self._cup])
        self._kuka.reset()
        p.stepSimulation()
        self._observation, info = self.getExtendedObservation()
        return np.array(self._observation), info

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        if self._image_output:  # for speeding up test, image output can be turned off
            camEyePos = [0.55, 0, 0]
            distance = 0.8
            pitch = -60
            yaw = 180
            roll = 0
            upAxisIndex = 2
            nearPlane = 0.01
            farPlane = 1000
            fov = 45
            viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
            projMatrix = p.computeProjectionMatrixFOV(fov, 1, nearPlane, farPlane, physicsClientId=0)

            img_arr = p.getCameraImage(width=self._width,
                                       height=self._height,
                                       viewMatrix=viewMat,
                                       projectionMatrix=projMatrix)
            rgb = img_arr[2]

            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                depth_buffer = img_arr[3].reshape(self._height, self._width)
                self._observation[3] = np.round(255*farPlane*nearPlane /
                                                ((farPlane-(farPlane-nearPlane)*depth_buffer)*1.1))

            elif self._mode == 'de':
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)
                img_arr2 = p.getCameraImage(width=self._width,
                                            height=self._height,
                                            viewMatrix=viewMat2,
                                            projectionMatrix=projMatrix)
                rgb2 = img_arr2[2]
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                self._observation[3] = rgb2[:, :, 0]
                self._observation[4] = rgb2[:, :, 1]
                self._observation[5] = rgb2[:, :, 2]


            elif self._mode =='pc':
                
                observation = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)

                img_arr2 = p.getCameraImage(self._width,
                                            self._height,
                                            viewMat2,
                                            projMatrix)
                rgb2 = img_arr2[2]
                observation2 = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = np.array(viewMat2).reshape(4, 4).T

                px = np.tile(np.arange(self._width, dtype=np.float32)[None, :], (self._height, 1))
                py = np.tile(np.arange(self._height, dtype=np.float32)[:, None], (1, self._width))
                viewMat = np.array(viewMat).reshape(4, 4).T

                projMatrix = np.array(projMatrix).reshape(4, 4).T
                T = np.linalg.inv(viewMat)
                u = (px / (self._width - 1) - 0.5) * 2
                v = (-py / (self._height - 1) + 0.5) * 2
                depth_buffer = img_arr[3].reshape(self._height, self._width)

                d = 2 * depth_buffer - 1
                z1 = -projMatrix[2][3] / (projMatrix[2][2] + d)
                x1 = u / projMatrix[0][0] * (-z1)
                y1 = v / projMatrix[1][1] * (-z1)

                observation[:, :, 0] = T[0][0] * x1 + T[0][1] * y1 + T[0][2] * z1 + T[0][3]
                observation[:, :, 1] = T[1][0] * x1 + T[1][1] * y1 + T[1][2] * z1 + T[1][3]
                observation[:, :, 2] = T[2][0] * x1 + T[2][1] * y1 + T[2][2] * z1 + T[2][3]
                observation[:, :, 3] = rgb[:, :, 0] / 255
                observation[:, :, 4] = rgb[:, :, 1] / 255
                observation[:, :, 5] = rgb[:, :, 2] / 255

                T2 = np.linalg.inv(viewMat2)
                depth_buffer2 = img_arr2[3].reshape(self._height, self._width)
                d2 = 2 * depth_buffer2 - 1

                z2 = -projMatrix[2][3] / (projMatrix[2][2] + d2)
                x2 = u / projMatrix[0][0] * (-z2)
                y2 = v / projMatrix[1][1] * (-z2)

                observation2[:, :, 0] = T2[0][0] * x2 + T2[0][1] * y2 + T2[0][2] * z2 + T2[0][3]
                observation2[:, :, 1] = T2[1][0] * x2 + T2[1][1] * y2 + T2[1][2] * z2 + T2[1][3]
                observation2[:, :, 2] = T2[2][0] * x2 + T2[2][1] * y2 + T2[2][2] * z2 + T2[2][3]
                observation2[:, :, 3] = rgb2[:, :, 0] / 255
                observation2[:, :, 4] = rgb2[:, :, 1] / 255
                observation2[:, :, 5] = rgb2[:, :, 2] / 255


                npoints = 2048
                sample_observation = np.r_[observation[observation[:, :, 2] > 0.0034], observation2[observation2[:, :, 2] > 0.0034]]
                sample_observation = sample_observation[sample_observation[:, 2] < 0.3]
                a, b = sample_observation.shape
               
                self._observation = np.zeros((npoints, 6), dtype=np.float32)
                if a > npoints:
                    choice = np.random.choice(a, npoints, replace=True, p=None)
                    self._observation[:, 0:3] = pc_normalize(sample_observation[choice, 0:3])
                    self._observation[:, 3:] = sample_observation[choice, 3:]
                if a < npoints:
                    choice = np.random.choice(a, npoints - a, replace=True, p=None)
                    self._observation[:, 0:3] = pc_normalize(
                        np.r_[sample_observation[:, 0:3], sample_observation[choice, 0:3]])
                    self._observation[:, 3:] = np.r_[sample_observation[:, 3:], sample_observation[choice, 3:]]
                if a == npoints:
                    self._observation[:, 0:3] = pc_normalize(sample_observation[:, 0:3])
                    self._observation[:, 3:] = sample_observation[:, 3:]


                self._observation = self._observation.T
             

        else:
            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
            elif self._mode == 'de' or self._mode == 'pc':
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
        additional_info = self._kuka.getObservation()     #末端执行器位置方向，爪子力，角度   4
        blockPos, blockQuat = p.getBasePositionAndOrientation(self._block)
        cupPos, cupQuat = p.getBasePositionAndOrientation(self._cup)
        blockEuler = p.getEulerFromQuaternion(blockQuat)
        cupEuler = p.getEulerFromQuaternion(cupQuat)
        additional_info.extend(list(blockPos))
        additional_info.extend(list(blockEuler))
        additional_info.extend(list(cupPos))
        additional_info.extend(list(cupEuler))
        additional_info = np.array(additional_info, dtype=np.float32)
        return self._observation, additional_info

    def step(self, action):
        action = np.clip(action, -1, 1)
        dv = 0.008
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        da = action[3] * 0.05
        df = action[4] * 0.1
        realAction = [dx, dy, dz, da, df]
        for i in range(self._actionRepeat):
            self._kuka.applyAction(realAction)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
       
        self._observation, info = self.getExtendedObservation()
       
        done, reward = self._reward()
        return np.array(self._observation), reward, done, info

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        rgb_array, _ = self.getExtendedObservation()
        return rgb_array

    def _reward(self):
        blockPos, blockOrn = p.getBasePositionAndOrientation(self._block)
        cupPos, cupOrn = p.getBasePositionAndOrientation(self._cup)
        cupEuler = p.getEulerFromQuaternion(cupOrn)
        if abs(cupEuler[0]) > 1 or abs(cupEuler[1]) > 1:
            return True, 0.0
        dist = np.sqrt((blockPos[0] - cupPos[0]) ** 2 + (blockPos[1] - cupPos[1]) ** 2)
        if dist < 0.01 and blockPos[2] - cupPos[2] < 0.05 and abs(cupEuler[0]) < 0.2 and abs(cupEuler[1]) < 0.2:
            return True, 1.0
        return False, 0.0


class KukaCamGymEnv2(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot="models/",
                 actionRepeat=3,
                 isEnableSelfCollision=True,
                 renders=False,
                 image_output=True,
                 mode='de',
                 width=128):
        print('******************************************\n'
              'Create Environment: \n'
              '     render?:', renders, ';\n'
              '     output image?:', image_output, ';\n'
              '     mode:', mode, ';\n'
              '     image_width:', width, '\n'
              '******************************************')
        self.seed()
        self._timeStep = 0.02
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._renders = renders
        self._image_output = image_output
        self._mode = mode
        self._width = width
        self._height = self._width
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.0, 230, -40, [0.55, 0, 0])
        else:
            p.connect(p.DIRECT)
        self._action_dim = 5
        self._action_bound = 1
        action_high = np.array([self._action_bound] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(6 if self._mode == 'de' else 4, self._height, self._width),
                                            dtype=np.uint8)
        self.viewer = None
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)
        p.loadURDF(self._urdfRoot+"floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF(self._urdfRoot+"table_collision/table.urdf", [0.6, 0, -0.625], useFixedBase=True)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._block = p.loadURDF(self._urdfRoot+"box_green.urdf")
        self._block2 = p.loadURDF(self._urdfRoot+"box_purple.urdf")
        self._range = 0.15

    def reset(self):
        collision = True
        while collision:
            set_pose(self._block, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                             np.random.uniform(-self._range, self._range), 0.025),
                                       [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            set_pose(self._block2, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                              np.random.uniform(-self._range, self._range), 0.025),
                                        [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            collision = check_pairwise_collisions([self._block, self._block2])
        self._kuka.reset()
        p.stepSimulation()
        self._observation, info = self.getExtendedObservation()
        return np.array(self._observation), info

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        if self._image_output:  # for speeding up test, image output can be turned off
            camEyePos = [0.55, 0, 0]
            distance = 0.8
            pitch = -60
            yaw = 180
            roll = 0
            upAxisIndex = 2
            nearPlane = 0.01
            farPlane = 1000
            fov = 45
            viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
            projMatrix = p.computeProjectionMatrixFOV(fov, 1, nearPlane, farPlane, physicsClientId=0)

            img_arr = p.getCameraImage(width=self._width,
                                       height=self._height,
                                       viewMatrix=viewMat,
                                       projectionMatrix=projMatrix)
            rgb = img_arr[2]
            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                depth_buffer = img_arr[3].reshape(self._height, self._width)
                self._observation[3] = np.round(255*farPlane*nearPlane /
                                                ((farPlane-(farPlane-nearPlane)*depth_buffer)*1.1))

            elif self._mode == 'de':
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)
                img_arr2 = p.getCameraImage(width=self._width,
                                            height=self._height,
                                            viewMatrix=viewMat2,
                                            projectionMatrix=projMatrix)
                rgb2 = img_arr2[2]
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                self._observation[3] = rgb2[:, :, 0]
                self._observation[4] = rgb2[:, :, 1]
                self._observation[5] = rgb2[:, :, 2]

            elif self._mode == 'pc':
            
                observation = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)

                img_arr2 = p.getCameraImage(self._width,
                                            self._height,
                                            viewMat2,
                                            projMatrix)
                rgb2 = img_arr2[2]
                observation2 = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = np.array(viewMat2).reshape(4, 4).T

                px = np.tile(np.arange(self._width, dtype=np.float32)[None, :], (self._height, 1))
                py = np.tile(np.arange(self._height, dtype=np.float32)[:, None], (1, self._width))
                viewMat = np.array(viewMat).reshape(4, 4).T

                projMatrix = np.array(projMatrix).reshape(4, 4).T
                T = np.linalg.inv(viewMat)
                u = (px / (self._width - 1) - 0.5) * 2
                v = (-py / (self._height - 1) + 0.5) * 2
                depth_buffer = img_arr[3].reshape(self._height, self._width)

                d = 2 * depth_buffer - 1
                z1 = -projMatrix[2][3] / (projMatrix[2][2] + d)
                x1 = u / projMatrix[0][0] * (-z1)
                y1 = v / projMatrix[1][1] * (-z1)

                observation[:, :, 0] = T[0][0] * x1 + T[0][1] * y1 + T[0][2] * z1 + T[0][3]
                observation[:, :, 1] = T[1][0] * x1 + T[1][1] * y1 + T[1][2] * z1 + T[1][3]
                observation[:, :, 2] = T[2][0] * x1 + T[2][1] * y1 + T[2][2] * z1 + T[2][3]
                observation[:, :, 3] = rgb[:, :, 0] / 255
                observation[:, :, 4] = rgb[:, :, 1] / 255
                observation[:, :, 5] = rgb[:, :, 2] / 255

                T2 = np.linalg.inv(viewMat2)
                depth_buffer2 = img_arr2[3].reshape(self._height, self._width)
                d2 = 2 * depth_buffer2 - 1

                z2 = -projMatrix[2][3] / (projMatrix[2][2] + d2)
                x2 = u / projMatrix[0][0] * (-z2)
                y2 = v / projMatrix[1][1] * (-z2)

                observation2[:, :, 0] = T2[0][0] * x2 + T2[0][1] * y2 + T2[0][2] * z2 + T2[0][3]
                observation2[:, :, 1] = T2[1][0] * x2 + T2[1][1] * y2 + T2[1][2] * z2 + T2[1][3]
                observation2[:, :, 2] = T2[2][0] * x2 + T2[2][1] * y2 + T2[2][2] * z2 + T2[2][3]
                observation2[:, :, 3] = rgb2[:, :, 0] / 255
                observation2[:, :, 4] = rgb2[:, :, 1] / 255
                observation2[:, :, 5] = rgb2[:, :, 2] / 255


                npoints = 1024
                sample_observation = np.r_[observation[observation[:, :, 2] > 0.0034], observation2[observation2[:, :, 2] > 0.0034]]
                sample_observation = sample_observation[sample_observation[:, 2] < 0.3]
                a, b = sample_observation.shape
               
                self._observation = np.zeros((npoints, 6), dtype=np.float32)
                if a > npoints:
                    choice = np.random.choice(a, npoints, replace=True, p=None)
                    self._observation[:, 0:3] = pc_normalize(sample_observation[choice, 0:3])
                    self._observation[:, 3:] = sample_observation[choice, 3:]
                if a < npoints:
                    choice = np.random.choice(a, npoints - a, replace=False, p=None)
                    self._observation[:, 0:3] = pc_normalize(
                        np.r_[sample_observation[:, 0:3], sample_observation[choice, 0:3]])
                    self._observation[:, 3:] = np.r_[sample_observation[:, 3:], sample_observation[choice, 3:]]
                if a == npoints:
                    self._observation[:, 0:3] = pc_normalize(sample_observation[:, 0:3])
                    self._observation[:, 3:] = sample_observation[:, 3:]


                self._observation = self._observation.T
                


        else:
            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
            elif self._mode == 'de' or self._mode== 'pc':
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
        additional_info = self._kuka.getObservation()
        blockPos, blockQuat = p.getBasePositionAndOrientation(self._block)
        block2Pos, block2Quat = p.getBasePositionAndOrientation(self._block2)
        blockEuler = p.getEulerFromQuaternion(blockQuat)
        block2Euler = p.getEulerFromQuaternion(block2Quat)
        additional_info.extend(list(blockPos))
        additional_info.extend(list(blockEuler))
        additional_info.extend(list(block2Pos))
        additional_info.extend(list(block2Euler))
        additional_info = np.array(additional_info, dtype=np.float32)
        return self._observation, additional_info

    def step(self, action):
        action = np.clip(action, -1, 1)
        dv = 0.008
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        da = action[3] * 0.05
        df = action[4] * 0.1
        realAction = [dx, dy, dz, da, df]
        for i in range(self._actionRepeat):
            self._kuka.applyAction(realAction)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        self._observation, info = self.getExtendedObservation()
        done, reward = self._reward()
        return np.array(self._observation), reward, done, info

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        rgb_array, _ = self.getExtendedObservation()
        return rgb_array

    def _reward(self):
        blockPos, blockOrn = p.getBasePositionAndOrientation(self._block)
        block2Pos, block2Orn = p.getBasePositionAndOrientation(self._block2)
        dist = np.sqrt((blockPos[0] - block2Pos[0]) ** 2 + (blockPos[1] - block2Pos[1]) ** 2)
        if dist < 0.02 and blockPos[2] < 0.076:
            return True, 1.0
        return False, 0.0


class KukaCamGymEnv3(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot="models/",
                 actionRepeat=3,
                 isEnableSelfCollision=True,
                 renders=False,
                 image_output=True,
                 mode='de',
                 width=128):
        print('******************************************\n'
              'Create Environment: \n'
              '     render?:', renders, ';\n'
              '     output image?:', image_output, ';\n'
              '     mode:', mode, ';\n'
              '     image_width:', width, '\n'
              '******************************************')
        self.seed()
        self._timeStep = 0.02
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._renders = renders
        self._image_output = image_output
        self._mode = mode
        self._width = width
        self._height = self._width
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.0, 230, -40, [0.55, 0, 0])
        else:
            p.connect(p.DIRECT)
        self._action_dim = 5
        self._action_bound = 1
        action_high = np.array([self._action_bound] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(6 if self._mode == 'de' else 4, self._height, self._width),
                                            dtype=np.uint8)
        self.viewer = None
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)
        p.loadURDF(self._urdfRoot+"floor.urdf", [0, 0, -0.625], useFixedBase=True)
        p.loadURDF(self._urdfRoot+"table_collision/table.urdf", [0.6, 0, -0.625], useFixedBase=True)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._cups = p.loadURDF(self._urdfRoot+"cup/cup_small.urdf")
        self._cup = p.loadURDF(self._urdfRoot+"cup/cup.urdf")
        self._range = 0.15

    def reset(self):
        collision = True
        while collision:
            set_pose(self._cups, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                            np.random.uniform(-self._range, self._range), 0.0745),
                                      [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            set_pose(self._cup, Pose(Point(np.random.uniform(-self._range + 0.55, self._range + 0.55),
                                           np.random.uniform(-self._range, self._range), 0.0745),
                                     [0, 0, np.random.uniform(-np.pi / 4, np.pi / 4)]))
            collision = check_pairwise_collisions([self._cups, self._cup])
        self._kuka.reset()
        p.stepSimulation()
        self._observation, info = self.getExtendedObservation()
        return np.array(self._observation), info

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        if self._image_output:  # for speeding up test, image output can be turned off
            camEyePos = [0.55, 0, 0]
            distance = 0.8
            pitch = -60
            yaw = 180
            roll = 0
            upAxisIndex = 2
            nearPlane = 0.01
            farPlane = 1000
            fov = 45
            viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
            projMatrix = p.computeProjectionMatrixFOV(fov, 1, nearPlane, farPlane, physicsClientId=0)

            img_arr = p.getCameraImage(width=self._width,
                                       height=self._height,
                                       viewMatrix=viewMat,
                                       projectionMatrix=projMatrix)
            rgb = img_arr[2]
            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                depth_buffer = img_arr[3].reshape(self._height, self._width)
                self._observation[3] = np.round(255*farPlane*nearPlane /
                                                ((farPlane-(farPlane-nearPlane)*depth_buffer)*1.1))

            elif self._mode == 'de':
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)
                img_arr2 = p.getCameraImage(width=self._width,
                                            height=self._height,
                                            viewMatrix=viewMat2,
                                            projectionMatrix=projMatrix)
                rgb2 = img_arr2[2]
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
                self._observation[0] = rgb[:, :, 0]
                self._observation[1] = rgb[:, :, 1]
                self._observation[2] = rgb[:, :, 2]
                self._observation[3] = rgb2[:, :, 0]
                self._observation[4] = rgb2[:, :, 1]
                self._observation[5] = rgb2[:, :, 2]

            elif self._mode == 'pc':
                observation = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, 0, pitch, roll, upAxisIndex)

                img_arr2 = p.getCameraImage(self._width,
                                            self._height,
                                            viewMat2,
                                            projMatrix)
                rgb2 = img_arr2[2]
                observation2 = np.zeros((self._height, self._width, 6), dtype=np.float32)
                viewMat2 = np.array(viewMat2).reshape(4, 4).T

                px = np.tile(np.arange(self._width, dtype=np.float32)[None, :], (self._height, 1))
                py = np.tile(np.arange(self._height, dtype=np.float32)[:, None], (1, self._width))
                viewMat = np.array(viewMat).reshape(4, 4).T

                projMatrix = np.array(projMatrix).reshape(4, 4).T
                T = np.linalg.inv(viewMat)
                u = (px / (self._width - 1) - 0.5) * 2
                v = (-py / (self._height - 1) + 0.5) * 2
                depth_buffer = img_arr[3].reshape(self._height, self._width)

                d = 2 * depth_buffer - 1
                z1 = -projMatrix[2][3] / (projMatrix[2][2] + d)
                x1 = u / projMatrix[0][0] * (-z1)
                y1 = v / projMatrix[1][1] * (-z1)

                observation[:, :, 0] = T[0][0] * x1 + T[0][1] * y1 + T[0][2] * z1 + T[0][3]
                observation[:, :, 1] = T[1][0] * x1 + T[1][1] * y1 + T[1][2] * z1 + T[1][3]
                observation[:, :, 2] = T[2][0] * x1 + T[2][1] * y1 + T[2][2] * z1 + T[2][3]
                observation[:, :, 3] = rgb[:, :, 0] / 255
                observation[:, :, 4] = rgb[:, :, 1] / 255
                observation[:, :, 5] = rgb[:, :, 2] / 255

                T2 = np.linalg.inv(viewMat2)
                depth_buffer2 = img_arr2[3].reshape(self._height, self._width)
                d2 = 2 * depth_buffer2 - 1

                z2 = -projMatrix[2][3] / (projMatrix[2][2] + d2)
                x2 = u / projMatrix[0][0] * (-z2)
                y2 = v / projMatrix[1][1] * (-z2)

                observation2[:, :, 0] = T2[0][0] * x2 + T2[0][1] * y2 + T2[0][2] * z2 + T2[0][3]
                observation2[:, :, 1] = T2[1][0] * x2 + T2[1][1] * y2 + T2[1][2] * z2 + T2[1][3]
                observation2[:, :, 2] = T2[2][0] * x2 + T2[2][1] * y2 + T2[2][2] * z2 + T2[2][3]
                observation2[:, :, 3] = rgb2[:, :, 0] / 255
                observation2[:, :, 4] = rgb2[:, :, 1] / 255
                observation2[:, :, 5] = rgb2[:, :, 2] / 255


                npoints = 1024
                sample_observation = np.r_[observation[observation[:, :, 2] > 0.0034], observation2[observation2[:, :, 2] > 0.0034]]
                sample_observation = sample_observation[sample_observation[:, 2] < 0.3]
                a, b = sample_observation.shape
               
                self._observation = np.zeros((npoints, 6), dtype=np.float32)
                if a > npoints:
                    choice = np.random.choice(a, npoints, replace=True, p=None)
                    self._observation[:, 0:3] = pc_normalize(sample_observation[choice, 0:3])
                    self._observation[:, 3:] = sample_observation[choice, 3:]
                if a < npoints:
                    choice = np.random.choice(a, npoints - a, replace=False, p=None)
                    self._observation[:, 0:3] = pc_normalize(
                        np.r_[sample_observation[:, 0:3], sample_observation[choice, 0:3]])
                    self._observation[:, 3:] = np.r_[sample_observation[:, 3:], sample_observation[choice, 3:]]
                if a == npoints:
                    self._observation[:, 0:3] = pc_normalize(sample_observation[:, 0:3])
                    self._observation[:, 3:] = sample_observation[:, 3:]


                self._observation = self._observation.T
        else:
            if self._mode == 'rgbd':
                self._observation = np.zeros((4, self._height, self._width), dtype=np.uint8)
            elif self._mode == 'de'or self._mode=='pc':
                self._observation = np.zeros((6, self._height, self._width), dtype=np.uint8)
        additional_info = self._kuka.getObservation()
        cupsPos, cupsQuat = p.getBasePositionAndOrientation(self._cups)
        cupPos, cupQuat = p.getBasePositionAndOrientation(self._cup)
        cupsEuler = p.getEulerFromQuaternion(cupsQuat)
        cupEuler = p.getEulerFromQuaternion(cupQuat)
        additional_info.extend(list(cupsPos))
        additional_info.extend(list(cupsEuler))
        additional_info.extend(list(cupPos))
        additional_info.extend(list(cupEuler))
        additional_info = np.array(additional_info, dtype=np.float32)
        return self._observation, additional_info

    def step(self, action):
        action = np.clip(action, -1, 1)
        dv = 0.008
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        da = action[3] * 0.05
        df = action[4] * 0.1
        realAction = [dx, dy, dz, da, df]
        for i in range(self._actionRepeat):
            self._kuka.applyAction(realAction)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        self._observation, info = self.getExtendedObservation()
        done, reward = self._reward()
        return np.array(self._observation), reward, done, info

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        rgb_array, _ = self.getExtendedObservation()
        return rgb_array

    def _reward(self):
        cupsPos, cupsOrn = p.getBasePositionAndOrientation(self._cups)
        cupPos, cupOrn = p.getBasePositionAndOrientation(self._cup)
        cupEuler = p.getEulerFromQuaternion(cupOrn)
        cupsEuler = p.getEulerFromQuaternion(cupsOrn)
        if abs(cupEuler[0]) > 1 or abs(cupEuler[1]) > 1:
            return True, 0.0
        if abs(cupsEuler[0]) > 1 or abs(cupsEuler[1]) > 1:
            return True, 0.0
        dist = np.sqrt((cupsPos[0] - cupPos[0]) ** 2 + (cupsPos[1] - cupPos[1]) ** 2)
        if dist < 0.01 and cupsPos[2] - cupPos[2] < 0.08 and abs(cupEuler[0]) < 0.2 and abs(cupEuler[1]) < 0.2:
            return True, 1.0
        return False, 0.0