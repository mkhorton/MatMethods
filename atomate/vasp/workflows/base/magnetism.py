# coding: utf-8

from atomate.vasp.powerups import add_tags, add_wf_metadata, add_common_powerups
from atomate.vasp.workflows.base.core import get_wf
from fireworks import Workflow, Firework
from atomate.vasp.firetasks.parse_outputs import MagneticDeformationToDB
from atomate.vasp.config import VASP_CMD, DB_FILE, ADD_WF_METADATA
import uuid

__author__ = "Matthew Horton, Joseph Montoya"
__version__ = "0.6"
__maintainer__ = "Matthew Horton"
__email__ = "mkhorton@lbl.gov"
__status__ = "Development"
__date__ = "March 2017"

def get_wf_magnetic_deformation(structure, c=None):
    """
    Minimal workflow to obtain magnetic deformation proxy, as
    defined by Bocarsly et al. 2017, doi: 10.1021/acs.chemmater.6b04729

    :param structure: input structure, must be structure with magnetic
    elements, such that pymatgen will initalize ferromagnetic input by
    default -- see MPRelaxSet.yaml for list of default elements
    :param c (dict): Workflow config dict, in the same format as in
    presets/core.py
    :return:
    """
    # TODO: when MagneticStructureAnalyzer is committed to pymatgen,
    # will add a default_magmoms kwarg to this method -mkhorton
    structure = structure.get_primitive_structure()
    uuid_tag = str(uuid.uuid4())  # unique tag to keep track of calculation

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    wf = get_wf(structure, "magnetic_deformation.yaml", vis=MPRelaxSet(structure),
                common_params={"vasp_cmd": vasp_cmd, "db_file": db_file})
    wf = add_tags(wf, [uuid_tag])
    wf = add_additional_fields_to_taskdocs(wf, {'magnetic_deformation_wf_uuid': uuid_tag})

    fw_analysis = Firework(MagneticDeformationToDB(db_file=db_file,
                                                   uuid=uuid_tag,
                                                   to_db=c.get("to_db", True)),
                           name="Calculate magnetic deformation")
    wf.append_wf(Workflow.from_Firework(fw_analysis), wf.leaf_fw_ids)

    wf = add_common_powerups(wf, c)

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf

# TODO: get_wf_magetic_orderings to go here, will calculate a series
# of AFM magnetic orderings and insert results into DB

if __name__ == "__main__":
    from fireworks import LaunchPad

    latt = Lattice.cubic(4.17)
    species = ["Ni", "O"]
    coords = [[0.00000, 0.00000, 0.00000],
              [0.50000, 0.50000, 0.50000]]
    NiO = Structure.from_spacegroup(225, latt, species, coords)
    wf = get_wf_magnetic_deformation(NiO)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
