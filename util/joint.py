import torch

class JointConverter:
    def __init__(self, num_joints):
        self.num_joints = num_joints
        self.joint_scale = 0

    def convert_for_training(self, joint):
        joint = self.normalize(joint)
        #joint = self._to_spherical_coord(joint)
        #joint = self._flatten(joint)

        return joint

    def convert_for_output(self, joint, no_unnormalize = False):
        #joint = self._unflatten(joint)
        #joint = self._to_cartesian_coord(joint)
        if no_unnormalize:
            return joint

        joint = self.unnormalize(joint)
        return joint

    def normalize(self, joint):
        """ make length between wrist and middle mcp as 1.0
            joint: (N, n_joint, 3)
        """

        wrist = joint[:, 0, :].unsqueeze(1)
        middle_mcp = joint[:, 9, :].unsqueeze(1)
        diff = wrist - middle_mcp
        scale = diff.norm(dim=2, keepdim = True)
        joint = joint / scale

        self.joint_scale = scale

        return joint

    def unnormalize(self, joint):
        """ change joint back to its original scale
            joint: (N, n_joint *3)
        """
        return joint * self.joint_scale

    def _to_spherical_coord(self, joint):
        X = joint[:,:,0]
        Y = joint[:,:,1]
        Z = joint[:,:,2]

        R = torch.sqrt(X**2 + Y**2 + Z**2)
        T = torch.acos(Z/R)
        P = torch.atan2(Y, X)

        joint = torch.stack([R, T, P], dim = 2)
        return joint

    def _to_cartesian_coord(self, joint):
        R = joint[:,:,0]
        T = joint[:,:,1]
        P = joint[:,:,2]

        Z = R * torch.cos(T)
        XY = R * torch.sin(T)
        X = XY * torch.cos(P)
        Y = XY * torch.sin(P)

        joint = torch.stack([X,Y,Z], dim = 2)
        return joint

    def _flatten(self, joint):
        if joint is None:
            return joint
        joint = joint.view(-1, 3*self.num_joints)
        return joint

    def _unflatten(self, joint):
        return joint.view(-1, self.num_joints, 3)
