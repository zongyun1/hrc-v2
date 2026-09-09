"""Restore Panda's authored opaque white material in the composed scene."""


def restore_panda_white(stage, robot_path):
    from pxr import Sdf, Usd, UsdShade

    robot = stage.GetPrimAtPath(robot_path)
    if not robot:
        raise ValueError(f'Robot prim not found: {robot_path}')
    replacements = []
    for prim in Usd.PrimRange(robot):
        if '/Franka/' not in str(prim.GetPath()):
            continue
        for relationship in prim.GetRelationships():
            if not relationship.GetName().startswith('material:binding'):
                continue
            targets = relationship.GetTargets()
            if not any(target.name == 'OmniSurface' for target in targets):
                continue
            geometry_path = str(prim.GetPath()).split('/geometry/')[0] + '/geometry'
            white_path = Sdf.Path(geometry_path + '/Looks/PlasticWhite')
            material = UsdShade.Material(stage.GetPrimAtPath(white_path))
            if not material:
                raise ValueError(f'Authored PlasticWhite material missing: {white_path}')
            relationship.SetTargets([white_path if target.name == 'OmniSurface' else target
                                     for target in targets])
            replacements.append({'prim': str(prim.GetPath()), 'material': str(white_path)})
    if not replacements:
        raise ValueError('Expected transparent Panda material bindings were not found')
    return {'mode': 'authored_PlasticWhite', 'rebound_subsets': len(replacements),
            'bindings': replacements}
