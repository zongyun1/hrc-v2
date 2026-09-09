"""Restore RoboCasa geom visibility and collision flags lost by MJCF conversion."""
import xml.etree.ElementTree as ET
import hashlib
import os
import shutil
from pathlib import Path


def prepare_converter_materials(xml_path):
    """Clamp visual shininess for USD roughness=1-shininess; keep source intact."""
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()
    changes = []
    for material in root.findall(".//material"):
        if "shininess" not in material.attrib:
            continue
        value = float(material.get("shininess"))
        clipped = min(1., max(0., value))
        if clipped != value:
            changes.append({"material": material.get("name"), "attribute": "shininess",
                            "source": value, "usd_import": clipped})
            material.set("shininess", str(clipped))
    if not changes:
        return xml_path, changes
    # Keep the copy beside the original so relative asset paths still resolve.
    converted_input = xml_path.with_name("scene_usd_materials.xml")
    converted_input.write_text(ET.tostring(root, encoding="unicode"))
    return converted_input, changes


def source_geom_rules(xml_path):
    rules = {}
    for geom in ET.parse(xml_path).getroot().findall(".//geom"):
        name = geom.get("name")
        if not name:
            continue
        rgba = [float(x) for x in geom.get("rgba", "1 1 1 1").split()]
        collision = int(geom.get("contype", "1")) != 0 or int(geom.get("conaffinity", "1")) != 0
        # RoboCasa separates display geoms (group 1) from collision proxies
        # (group 0). Transparent reset/bounding regions are semantic only.
        visible = int(geom.get("group", "0")) == 1 and rgba[-1] > 0
        rules[name] = {"collision": collision, "visible": visible}
    return rules


def apply_geometry_rules(kitchen, xml_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    rules = source_geom_rules(xml_path)
    found, disabled, hidden = set(), 0, 0
    for prim in Usd.PrimRange(kitchen):
        rule = rules.get(prim.GetName())
        if rule is None:
            continue
        found.add(prim.GetName())
        if not rule["visible"] and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
        if not rule["collision"]:
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                    disabled += 1
    return {"source_geoms": len(rules), "matched_geoms": len(found),
            "disabled_collision_prims": disabled, "hidden_proxy_prims": hidden,
            "unmatched_geoms": sorted(set(rules)-found)}


def repair_material_textures(kitchen, xml_path, texture_dir):
    """Avoid the converter's basename collisions (e.g. many different T_BC001.png files)."""
    from pxr import Usd, Sdf
    xml = ET.parse(xml_path).getroot()
    textures = {t.get("name"): t.get("file") for t in xml.findall("asset/texture") if t.get("file")}
    materials = {m.get("name"): textures[m.get("texture")] for m in xml.findall("asset/material")
                 if m.get("texture") in textures}
    texture_dir = Path(texture_dir)
    texture_dir.mkdir(parents=True, exist_ok=True)
    remapped = {}
    for prim in Usd.PrimRange(kitchen):
        if prim.GetTypeName() != "Material" or prim.GetName() not in materials:
            continue
        source = Path(materials[prim.GetName()])
        unique = hashlib.sha256(str(source).encode()).hexdigest()[:16] + "_" + source.name
        dest = texture_dir / unique
        if not dest.exists():
            shutil.copy2(source, dest)
        attr = prim.GetAttribute("inputs:DiffuseTexture")
        if attr:
            author_file = kitchen.GetStage().GetEditTarget().GetLayer().realPath
            texture_path = (os.path.relpath(dest.resolve(), Path(author_file).parent)
                            if author_file else str(dest.resolve()))
            attr.Set(Sdf.AssetPath(texture_path))
            remapped[prim.GetName()] = str(dest)
    return remapped
