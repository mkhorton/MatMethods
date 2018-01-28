# coding: utf-8

import os

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW
from fireworks import Workflow, Firework
from atomate.vasp.powerups import add_tags, add_additional_fields_to_taskdocs,\
    add_wf_metadata, add_common_powerups
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.firetasks.parse_outputs import MagneticDeformationToDB, MagneticOrderingsToDB

from pymatgen.transformations.advanced_transformations import MagOrderParameterConstraint, \
    MagOrderingTransformation
from pymatgen.analysis.local_env import MinimumDistanceNN

from atomate.utils.utils import get_logger
logger = get_logger(__name__)

from atomate.vasp.config import VASP_CMD, DB_FILE, ADD_WF_METADATA
from uuid import uuid4
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer, Ordering

__author__ = "Matthew Horton"
__maintainer__ = "Matthew Horton"
__email__ = "mkhorton@lbl.gov"
__status__ = "Development"
__date__ = "March 2017"

__magnetic_deformation_wf_version__ = 1.2
__magnetic_ordering_wf_version__ = 1.3

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def get_wf_magnetic_deformation(structure,
                                common_params=None,
                                vis=None):
    """
    Minimal workflow to obtain magnetic deformation proxy, as
    defined by Bocarsly et al. 2017, doi: 10.1021/acs.chemmater.6b04729

    :param structure: input structure, must be structure with magnetic
    elements, such that pymatgen will initalize ferromagnetic input by
    default -- see MPRelaxSet.yaml for list of default elements
    :param common_params (dict): Workflow config dict, in the same format
    as in presets/core.py
    :param vis: (VaspInputSet) A VaspInputSet to use for the first FW
    :return:
    """

    if not structure.is_ordered:
        raise ValueError("Please obtain an ordered approximation of the input structure.")

    structure = structure.get_primitive_structure(use_site_props=True)

    # using a uuid for book-keeping,
    # in a similar way to other workflows
    uuid = str(uuid4())

    c = {'vasp_cmd': VASP_CMD, 'db_file': DB_FILE}
    if common_params:
        c.update(common_params)

    wf = get_wf(structure, "magnetic_deformation.yaml",
                common_params=c, vis=vis)

    fw_analysis = Firework(
        MagneticDeformationToDB(
            db_file=DB_FILE,
            wf_uuid=uuid,
            to_db=c.get("to_db", True)),
        name="MagneticDeformationToDB")

    wf.append_wf(Workflow.from_Firework(fw_analysis), wf.leaf_fw_ids)

    wf = add_common_powerups(wf, c)

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    wf = add_additional_fields_to_taskdocs(wf, {
        'wf_meta': {
            'wf_uuid': uuid,
            'wf_name': 'magnetic_deformation',
            'wf_version': __magnetic_deformation_wf_version__
        }})

    return wf

class MagneticOrderingsWF:

    def __init__(self, structure, default_magmoms=None,
                 respect_input_magmoms="replace_all",
                 attempt_ferromagnetic=True,
                 attempt_antiferromagnetic=True,
                 attempt_ferrimagnetic_by_species=None,
                 attempt_ferrimagnetic_by_motif=None,
                 attempt_antiferromagnetic_by_motif=None,
                 num_orderings=8,
                 max_unique_sites=8,
                 max_cell_size=None,
                 timeout=5
                 ):
        """
        This workflow will try several different collinear
        magnetic orderings for a given input structure,
        and output a summary to a dedicated database
        collection, magnetic_orderings, in the supplied
        db_file.

        If the input structure has magnetic moments defined, it
        is possible to use these as a hint as to which elements are
        magnetic, otherwise magnetic elements will be guessed
        (this can be changed using default_magmoms kwarg).

        A brief description on how this workflow works:
            1. We make a note of the input structure, and then
               sanitize it (make it ferromagnetic, primitive)
            2. We gather information on which sites are likely
               magnetic, how many unique magnetic sites there
               are (whether one species is in several unique
               environments, e.g. tetrahedral/octahedra as Fe
               in a spinel)
            3. We generate ordered magnetic structures, first
               antiferromagnetic, and then, if appropriate,
               ferrimagnetic structures either by species or
               by environment -- this makes use of some new
               additions to MagOrderingTransformation to allow
               the spins of different species to be coupled together
               (e.g. all one species spin up, all other species spin
               down, with an overall order parameter of 0.5)
            4. For each ordered structure, we perform a relaxation
               and static calculation. Then an aggregation is performed
               which finds which ordering is the ground state (of
               those attempted in this specific workflow). For
               high-throughput studies, a dedicated builder is
               recommended.
            5. For book-keeping, a record is kept of whether the
               input structure is enumerated by the algorithm or
               not. This is useful when supplying the workflow with
               a magnetic structure obtained by experiment, to measure
               the performance of the workflow.


        :param structure: input structure, may be a generic crystal
    structure, or a structure with magnetic moments already
    defined (e.g. from experiment)
        :param default_magmoms: dict of magnetic elements
    to their initial magnetic moments in µB, generally these
    are chosen to be high-spin since they can relax to a low-spin
    configuration during a DFT electronic configuration
        :param respect_input_magmoms:
        :param attempt_ferromagnetic:
        :param attempt_antiferromagnetic:
        :param attempt_ferrimagnetic_by_species:
        :param attempt_ferrimagnetic_by_motif:
        :param attempt_antiferromagnetic_by_motif: (beta) whether or not
    to attempt antiferromagnetic orderingon just a subset of
    sites. This is only appropriate in some cases, use with caution.
    In general, an oxidation-state decorated structure and/or
    physical intuition can help in these cases.
        :param num_orderings: This is the number of each
    type of ordering to attempt. Since structures are grouped
    by symmetry, the number of returned structures will not
    necessarily be exactly num_orderings if several enumared
    structures have equivalent symmetries. Note this is also
    per strategy, so it will return num_orderings AFM orderings,
    but also num_orderings ferrimagnetic by motif, etc. if
    attempt_ferrimagnetic == True
        :param max_unique_sites: The max_cell_size to consider. If
    too large, enumeration will take a long time! By default will
    try a sensible value (between 4 and 1 depending on number of
    magnetic sites in primitive cell).
        :param max_cell_size:
        :param timeout:
        """

        self.structure = structure

        # decides how to process input structure, which sites are magnetic
        self.default_magmoms = default_magmoms
        self.respect_input_magmoms = respect_input_magmoms

        # different strategies to attempt, default is usually reasonable
        self.attempt_ferromagnetic = attempt_ferromagnetic
        self.attempt_antiferromagnetic = attempt_antiferromagnetic
        self.attempt_ferrimagnetic_by_motif = attempt_ferrimagnetic_by_motif
        self.attempt_ferrimagnetic_by_species = attempt_ferrimagnetic_by_species
        self.attempt_antiferromagnetic_by_motif = attempt_antiferromagnetic_by_motif

        # other settings
        self.num_orderings = num_orderings
        self.max_unique_sites = max_unique_sites
        self.max_cell_size = max_cell_size
        self.timeout = timeout

        # our magnetically ordered structures will be
        # stored here once generated and also store which
        # transformation created them, this is used for
        # book-keeping/user interest, and
        # is be a list of strings in ("fm", "afm",
        # "ferrimagnetic_by_species", "ferrimagnetic_by_motif",
        # "afm_by_motif", "input_structure")
        self.ordered_structures, self.ordered_structure_origins = [], []

        formula = structure.composition.reduced_formula

        # to process disordered magnetic structures, first make an
        # ordered approximation
        if not structure.is_ordered:
            raise ValueError("Please obtain an ordered approximation of the "
                             "input structure ({}).".format(formula))

        # CollinearMagneticStructureAnalyzer is used throughout:
        # it can tell us whether the input is itself collinear (if not,
        # this workflow is not appropriate), and has many convenience
        # methods e.g. magnetic structure matching, etc.
        self.input_analyzer = CollinearMagneticStructureAnalyzer(structure,
                                                                 default_magmoms=default_magmoms,
                                                                 overwrite_magmom_mode="none")


        # this workflow enumerates structures with different combinations
        # of up and down spin and does not include spin-orbit coupling:
        # if your input structure has vector magnetic moments, this
        # workflow is not appropriate
        if not self.input_analyzer.is_collinear:
            raise ValueError("Input structure ({}) is non-collinear.".format(formula))

        self.sanitized_structure = self._sanitize_input_structure(structure)

        # we will first create a set of transformations
        # and then apply them to our input structure
        self.transformations = self._generate_transformations(self.sanitized_structure)
        self._generate_ordered_structures(self.sanitized_structure, self.transformations)


    def _sanitize_input_structure(self, input_structure):

        input_structure = input_structure.copy()

        # remove any annotated spin
        input_structure.remove_spin()

        # sanitize input structure: first make primitive ...
        input_structure = input_structure.get_primitive_structure(use_site_props=False)

        # ... and strip out existing magmoms, which can cause conflicts
        # with later transformations otherwise since sites would end up
        # with both magmom site properties and Specie spins defined
        if 'magmom' in input_structure.site_properties:
            input_structure.remove_site_property('magmom')

        return input_structure

    def _generate_transformations(self, structure):

        formula = structure.composition.reduced_formula
        transformations = {}

        # analyzer is used to obtain information on sanitized input
        analyzer = CollinearMagneticStructureAnalyzer(structure,
                                                      default_magmoms=self.default_magmoms,
                                                      overwrite_magmom_mode="replace_all")

        # now we can begin to generate our magnetic orderings
        logger.info("Generating magnetic orderings for {}".format(formula))

        mag_species_spin = analyzer.magnetic_species_and_magmoms
        types_mag_species = analyzer.types_of_magnetic_specie
        types_mag_elements = {sp.symbol for sp in types_mag_species}
        num_mag_sites = analyzer.number_of_magnetic_sites
        num_unique_sites = analyzer.number_of_unique_magnetic_sites()

        # enumerations become too slow as number of unique sites (and thus
        # permutations) increase, 8 is a soft limit, this can be increased
        # but do so with care
        if num_unique_sites > self.max_unique_sites:
            raise ValueError("Too many magnetic sites to sensibly perform enumeration.")

        # maximum cell size to consider: as a rule of thumb, if the primitive cell
        # contains a large number of magnetic sites, perhaps we only need to enumerate
        # within one cell, whereas on the other extreme if the primitive cell only
        # contains a single magnetic site, we have to create larger supercells
        max_cell_size = self.max_cell_size if self.max_cell_size else max(1, int(4 / num_mag_sites))
        logger.info("Max cell size set to {}".format(max_cell_size))

        # when enumerating ferrimagnetic structures, it's useful to detect
        # co-ordination numbers on the magnetic sites, since different
        # local environments can result in different magnetic order
        # (e.g. inverse spinels)
        # to do this more exhaustively, could also order per Wyckoff site
        nn = MinimumDistanceNN()
        cns = [nn.get_cn(structure, n) for n in range(len(structure))]
        is_magnetic_sites = [True if site.specie in types_mag_species
                             else False for site in structure]
        # we're not interested in co-ordination numbers for sites
        # that we don't think are magnetic, set these to zero
        cns = [cn if is_magnetic_site else 0
               for cn, is_magnetic_site in zip(cns, is_magnetic_sites)]
        structure.add_site_property('cn', cns)
        unique_cns = set(cns) - {0}

        # if user doesn't specifically request ferrimagnetic orderings,
        # we apply a heuristic as to whether to attempt them or not
        if self.attempt_ferrimagnetic_by_motif is None \
                and len(unique_cns) > 1 and len(types_mag_species) == 1:
            self.attempt_ferrimagnetic_by_motif = True

        if self.attempt_ferrimagnetic_by_species is None \
                and len(types_mag_species) > 1:
            self.attempt_ferrimagnetic_by_species = True

        # we start with a ferromagnetic ordering
        if self.attempt_ferromagnetic:
            fm_structure = analyzer.get_ferromagnetic_structure()
            # store magmom as spin property, to be consistent with output from
            # other transformations
            fm_structure.add_spin_by_site(fm_structure.site_properties['magmom'])
            fm_structure.remove_site_property('magmom')

            # we now have our first magnetic ordering...
            self.ordered_structures.append(fm_structure)
            self.ordered_structure_origins.append("fm")

        # ...to which we can add simple AFM cases first...
        if self.attempt_antiferromagnetic:
            constraint = MagOrderParameterConstraint(
                0.5,
                # TODO: this list(map(str...)) can probably be removed
                species_constraints=list(map(str, types_mag_species))
            )

            trans = MagOrderingTransformation(mag_species_spin,
                                              order_parameter=[constraint],
                                              max_cell_size=max_cell_size,
                                              timeout=self.timeout)
            transformations["afm"] = trans

        # ...and then we also try ferrimagnetic orderings by motif if a
        # single magnetic species is present...
        if self.attempt_ferrimagnetic_by_motif:

            # these orderings are AFM on one local environment, and FM on the rest
            for cn in unique_cns:
                constraints = [
                    MagOrderParameterConstraint(
                        0.5,
                        site_constraint_name='cn',
                        site_constraints=cn
                    ),
                    MagOrderParameterConstraint(
                        1.0,
                        site_constraint_name='cn',
                        site_constraints=list(unique_cns - {cn})
                    )
                ]

                trans = MagOrderingTransformation(mag_species_spin,
                                                  order_parameter=constraints,
                                                  max_cell_size=max_cell_size,
                                                  timeout=self.timeout)

                transformations["ferrimagnetic_by_motif"] = trans

        # and also try ferrimagnetic when there are multiple magnetic species
        if self.attempt_ferrimagnetic_by_species:

            for sp in types_mag_species:
                constraints = [
                    MagOrderParameterConstraint(
                        0.5,
                        species_constraints=str(sp)
                    ),
                    MagOrderParameterConstraint(
                        1.0,
                        species_constraints=list(map(str, set(types_mag_species) - {sp}))
                    )
                ]

                trans = MagOrderingTransformation(mag_species_spin,
                                                  order_parameter=constraints,
                                                  max_cell_size=max_cell_size,
                                                  timeout=self.timeout)

                transformations["ferrimagnetic_by_species"] = trans

        # ...and finally, we can try orderings that are AFM on one local
        # environment, and non-magnetic on the rest -- this is less common
        # but unless explicitly attempted, these states are unlikely to be found
        if self.attempt_antiferromagnetic_by_motif:

            logger.warning("Selectively performing antiferromagnetic orderings by "
                           "motif is in beta.")
            # TODO: ensure that zero magmom sites are not being overwritten by VaspInputSet
            # apply spin=0 if this is an issue

            for cn in unique_cns:
                constraints = [
                    MagOrderParameterConstraint(
                        0.5,
                        site_constraint_name='cn',
                        site_constraints=cn
                    )
                ]

                trans = MagOrderingTransformation(mag_species_spin,
                                                  order_parameter=constraints,
                                                  max_cell_size=max_cell_size,
                                                  timeout=self.timeout)

                transformations["afm_by_motif"] = trans

        return transformations

    def _generate_ordered_structures(self, sanitized_input_structure, transformations):

        ordered_structures = self.ordered_structures
        ordered_structures_origins = self.ordered_structure_origins

        # utility function to combine outputs from several transformations
        def _add_structures(ordered_structures, ordered_structures_origins,
                            structures_to_add, origin=""):
            """
            Transformations with return_ranked_list can return either
            just Structures or dicts (or sometimes lists!) -- until this
            is fixed, we use this function to concat structures given
            by the transformation.
            """
            if structures_to_add:
                # type conversion
                if isinstance(structures_to_add, Structure):
                    structures_to_add = [structures_to_add]
                structures_to_add = [s["structure"] if isinstance(s, dict)
                                     else s for s in structures_to_add]
                # concatenation
                ordered_structures += structures_to_add
                ordered_structures_origins += [origin]*len(structures_to_add)
                logger.info('Adding {} ordered structures: {}'.format(len(structures_to_add),
                                                                      origin))


            return ordered_structures, ordered_structures_origins

        for origin, trans in self.transformations.items():
            structures_to_add = trans.apply_transformation(self.sanitized_structure,
                                                           return_ranked_list=self.num_orderings)
            ordered_structures, \
            ordered_structures_origins = _add_structures(ordered_structures,
                                                         ordered_structures_origins,
                                                         structures_to_add,
                                                         origin=origin)

        # in case we've introduced duplicates, let's remove them
        structures_to_remove = []
        for idx, ordered_structure in enumerate(ordered_structures):
            if idx not in structures_to_remove:
                duplicate_analyzer = CollinearMagneticStructureAnalyzer(ordered_structure,
                                                                        overwrite_magmom_mode="none")
                matches = [duplicate_analyzer.matches_ordering(s)
                           for s in ordered_structures]
                structures_to_remove += [match_idx for match_idx, match in enumerate(matches)
                                         if (match and idx != match_idx)]

        if len(structures_to_remove):
            logger.info(
                'Removing {} duplicate ordered structures'.format(len(structures_to_remove)))
            ordered_structures = [s for idx, s in enumerate(ordered_structures)
                                  if idx not in structures_to_remove]
            ordered_structures_origins = [o for idx, o in enumerate(ordered_structures_origins)
                                          if idx not in structures_to_remove]

        # if our input structure isn't in our generated structures,
        # let's add it manually and also keep a note of which structure
        # is our input: this is mostly for book-keeping/benchmarking
        self.input_index = None
        if self.input_analyzer.ordering != Ordering.NM:
            matches = [self.input_analyzer.matches_ordering(s) for s in ordered_structures]
            if not any(matches):
                ordered_structures.append(self.input_analyzer.structure)
                ordered_structures_origins.append("input")
                logger.info("Input structure not present in enumerated structures, adding...")
            else:
                logger.info("Input structure was found in enumerated "
                            "structures at index {}".format(matches.index(True)))
                self.input_index = matches.index(True)

        self.ordered_structures = ordered_structures
        self.ordered_structure_origins = ordered_structures_origins

        return

    def get_wf(self, vasp_cmd=VASP_CMD,
                     db_file=DB_FILE,
                     vasp_input_set_kwargs=None,
                     optimize_fw=None,
                     static_fw=None,
                     perform_bader=True,
                     num_orderings_limit=8):
        """

        :param vasp_cmd: as elsewhere in atomate
        :param db_file: as elsewhere in atomate (calculations
        will be added to "tasks" collection, and summary of
        orderings to "magnetic_orderings" collection)
        :param vasp_input_set_kwargs: kwargs to pass to the
    vasp input set, the default is `{'user_incar_settings':
    {'ISYM': 0, 'LASPH': True}`
        :param optimize_fw:
        :param static_fw:
        :param perform_bader: Perform Bader analysis (can
    provide a more robust measure of per-atom magnetic moments).
    Requires bader binary to be in path.
        :return:
        """

        fws = []
        analysis_parents = []

        # trim total number of orderings (useful in high-throughput context)
        # this is somewhat course, better to reduce num_orderings kwarg and/or
        # change enumeration strategies
        ordered_structures = self.ordered_structures
        ordered_structure_origins = self.ordered_structure_origins
        if num_orderings_limit and len(self.ordered_structures) > num_orderings_limit:
            ordered_structures = self.ordered_structures[0:num_orderings_limit]
            ordered_structure_origins = self.ordered_structure_origins[0:num_orderings_limit]
            logger.warning("Number of ordered structures exceeds hard limit, "
                           "removing last {} structures.".format(len(self.ordered_structures)-
                                                                 len(ordered_structures)))
            # always make sure input structure is included
            if self.input_index and self.input_index > num_orderings_limit:
                ordered_structures.append(self.ordered_structures[self.input_index])
                ordered_structure_origins.append(self.ordered_structure_origins[self.input_index])

        for idx, ordered_structure in enumerate(ordered_structures):

            analyzer = CollinearMagneticStructureAnalyzer(ordered_structure)

            name = "ordering {} {} -".format(idx, analyzer.ordering.value)

            # get keyword arguments for VaspInputSet
            relax_vis_kwargs = {'user_incar_settings': {'ISYM': 0, 'LASPH': True}}
            if vasp_input_set_kwargs:
                relax_vis_kwargs.update(vasp_input_set_kwargs)

            vis = MPRelaxSet(ordered_structure, **relax_vis_kwargs)

            # relax
            fws.append(OptimizeFW(ordered_structure, vasp_input_set=vis,
                                  vasp_cmd=vasp_cmd, db_file=db_file,
                                  max_force_threshold=0.25, # TODO: decrease
                                  half_kpts_first_relax=False,
                                  name=name + " optimize"))

            # static
            fws.append(StaticFW(ordered_structure, vasp_cmd=vasp_cmd,
                                db_file=db_file,
                                name=name + " static",
                                vasp_to_db_kwargs={"perform_bader": perform_bader},
                                prev_calc_loc=True, parents=fws[-1]))

            analysis_parents.append(fws[-1])

        uuid = str(uuid4())
        fw_analysis = Firework(MagneticOrderingsToDB(db_file=db_file,
                                                     wf_uuid=uuid,
                                                     auto_generated=False,
                                                     name="MagneticOrderingsToDB",
                                                     parent_structure=self.sanitized_structure,
                                                     strategy=vasp_input_set_kwargs,
                                                     origins=ordered_structure_origins,
                                                     input_index=self.input_index,
                                                     perform_bader=perform_bader),
                               name="Magnetic Orderings Analysis", parents=analysis_parents)
        fws.append(fw_analysis)

        formula = self.sanitized_structure.composition.reduced_formula
        wf_name = "{} - magnetic orderings".format(formula)
        wf = Workflow(fws, name=wf_name)

        wf = add_additional_fields_to_taskdocs(wf, {
            'wf_meta': {
                'wf_uuid': uuid,
                'wf_name': 'magnetic_orderings',
                'wf_version': __magnetic_ordering_wf_version__
            }})

        tag = "magnetic_orderings group: >>{}<<".format(uuid)
        wf = add_tags(wf, [tag, ordered_structure_origins])

        self._wf = wf

        return wf


if __name__ == "__main__":

    # for trying workflow

    from fireworks import LaunchPad

    latt = Lattice.cubic(4.17)
    species = ["Ni", "O"]
    coords = [[0.00000, 0.00000, 0.00000],
              [0.50000, 0.50000, 0.50000]]
    NiO = Structure.from_spacegroup(225, latt, species, coords)

    wf_deformation = get_wf_magnetic_deformation(NiO)

    wf_orderings = MagneticOrderingsWF(NiO).get_wf()

    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf_orderings)
    lpad.add_wf(wf_deformation)
