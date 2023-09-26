from highlevel_planning_py.sim.scene_base import SceneBase
from highlevel_planning_py.sim.cupboard import get_cupboard_info


class SceneMoveSkill(SceneBase):
    def __init__(self, world, paths, restored_objects=None):
        SceneBase.__init__(self, world, paths, restored_objects)

        if restored_objects is None:
            self.objects = dict()
            self.objects["cupboard"] = get_cupboard_info(
                paths["asset_dir"], pos=[0.0, 2.0, 0.0], orient=[0.0, 0.0, 0.0, 1.0]
            )
            self.add_objects()
