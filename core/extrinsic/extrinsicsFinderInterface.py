class ExtrinsicFinderInterface:
    def __init__(self):
        self.camera_mtx = None
        self.camera_dist = None
        self.theory_point = None
        pass

    def set_param(self, mtx=None, dist=None, theory_point=None):
        pass

    def get_extrinsic(self, gray_image=None) -> ([], []):
        pass
