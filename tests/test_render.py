import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from inkform.render import layout_document, strokes_from_offsets, render_page, export_document, _vary_letters, _vary_word_gaps, _adjust_slant
from inkform.schema import InkSettings, PageSettings


def test_pen_lift_does_not_connect_strokes():
    strokes = strokes_from_offsets([(0, 0, 0), (1, 1, 1), (50, 50, 0), (1, 1, 1)])
    assert len(strokes) == 2
    np.testing.assert_array_equal(strokes[0], [[0, 0], [1, 1]])
    np.testing.assert_array_equal(strokes[1], [[51, 51], [52, 52]])


def test_slant_adjustment_reduces_rightward_lean_preserves_joins_and_vertical_geometry():
    # Top of a right-leaning downstroke is right of its bottom; after a -45
    # degree shear this example is vertical. A coincident join stays joined.
    source=[np.array([[2.,0.],[0.,2.]]),np.array([[0.,2.],[1.,3.]])]
    before=[s.copy() for s in source]
    adjusted=_adjust_slant(source,-45)
    np.testing.assert_allclose(adjusted[0][0,0],adjusted[0][1,0],atol=1e-12)
    np.testing.assert_array_equal(adjusted[0][-1],adjusted[1][0])
    for original,copy,moved in zip(source,before,adjusted):
        np.testing.assert_array_equal(original,copy)
        np.testing.assert_array_equal(moved[:,1],original[:,1])


@pytest.mark.parametrize('angle',[-50,0,50])
def test_slant_adjustment_keeps_page_bounds_and_saved_trajectories(document,angle):
    original=document.model_dump_json()
    page=PageSettings(slant_adjustment_deg=angle,word_gap_min_mm=2.5)
    layout=layout_document(document,page)
    for rows in layout.pages:
        for line in rows:
            a=np.concatenate(line.strokes)
            assert a[:,0].min()>=page.margin_mm-1e-6 and a[:,0].max()<=210-page.margin_mm+1e-6
            assert a[:,1].min()>=page.margin_mm-1e-6 and a[:,1].max()<=297-page.margin_mm+1e-6
    assert document.model_dump_json()==original


def test_pagination_fits_every_stroke_inside_printable_area(document):
    document.lines *= 54
    page = PageSettings()
    layout = layout_document(document, page)
    assert len(layout.pages) == 3
    assert sum(map(len, layout.pages)) == 54
    for lines in layout.pages:
        for line in lines:
            points = np.concatenate(line.strokes)
            assert points[:, 0].min() >= page.margin_mm - 1e-6
            assert points[:, 0].max() <= 210 - page.margin_mm + 1e-6
            assert points[:, 1].min() >= page.margin_mm - 1e-6
            assert points[:, 1].max() <= 297 - page.margin_mm + 1e-6


def test_render_is_deterministic_and_ink_does_not_modify_trajectories(document):
    before = document.model_dump_json()
    page = PageSettings(paper="plain")
    lines = layout_document(document, page).pages[0]
    image_a = render_page(lines, page, InkSettings(), 96)
    image_b = render_page(lines, page, InkSettings(), 96)
    image_c = render_page(lines, page, InkSettings(color="#000000", roughness=0), 96)
    np.testing.assert_array_equal(image_a, image_b)
    assert not np.array_equal(image_a, image_c)
    assert document.model_dump_json() == before
    assert np.asarray(image_a).min() < 200


def test_export_has_a4_physical_size_and_resolution(document, tmp_path):
    document.lines *= 27
    manifest = export_document(document, PageSettings(), InkSettings(), tmp_path, 150)
    assert manifest["pages"] == 2
    reader = PdfReader(tmp_path / "inkform.pdf")
    assert len(reader.pages) == 2
    for page in reader.pages:
        assert abs(float(page.mediabox.width) - 595.2756) < 0.01
        assert abs(float(page.mediabox.height) - 841.8898) < 0.01
    with Image.open(tmp_path / "page-001.png") as image:
        assert image.size == (1240, 1754)
        # PNG stores integer pixels/metre, quantizing DPI by up to 0.0127.
        assert abs(image.info["dpi"][0] - 150) <= 0.0127


@pytest.mark.parametrize("height,spacing,margin", [(3,7,10), (9,10.5,40), (6.5,10,20)])
@pytest.mark.parametrize("individual",[False,True])
def test_full_relaxation_preserves_strokes_and_page_bounds(document,height,spacing,margin,individual):
    document.lines *= 54
    before = document.model_dump_json()
    settings = PageSettings(writing_height_mm=height,line_spacing_mm=spacing,margin_mm=margin,naturalness=1,letter_variation=1,
        width_variation=1,height_variation=1,slant_variation=1,rhythm_variation=1,word_spacing_variation=1,
        line_start_variation=1,line_end_variation=1,fit_lines_individually=individual,
        line_spacing_variation=1,line_tilt_variation=1,baseline_variation=1)
    layout = layout_document(document,settings)
    for page in layout.pages:
        for row,line in enumerate(page):
            points = np.concatenate(line.strokes)
            assert points[:,0].min() >= margin-1e-6
            assert points[:,0].max() <= 210-margin+1e-6
            assert points[:,1].min() >= margin+row*spacing-1e-6
            assert points[:,1].max() <= margin+(row+1)*spacing+1e-6
            assert [len(s) for s in line.strokes] == [len(s) for s in strokes_from_offsets(document.lines[0].offsets)]
    assert document.model_dump_json() == before


def test_one_wide_line_does_not_shrink_other_lines_when_individual_fit_is_enabled(document):
    wide=document.lines[0].model_copy(deep=True)
    wide.offsets=[(x*3,y,end) for x,y,end in wide.offsets]
    document.lines.append(wide)
    before=document.model_dump_json()
    settings=PageSettings(naturalness=0,letter_variation=0)
    shared=layout_document(document,settings).pages[0]
    individual=layout_document(document,settings.model_copy(update={'fit_lines_individually':True})).pages[0]
    height=lambda line:np.ptp(np.concatenate(line.strokes)[:,1])
    assert height(individual[0])>height(shared[0])*1.5
    assert height(individual[1])<=height(individual[0])
    assert document.model_dump_json()==before


def test_line_end_control_preserves_varied_starts_and_is_repeatable(document):
    document.lines[0].offsets=[(x*2,y,end) for x,y,end in document.lines[0].offsets]
    document.lines*=6
    settings=PageSettings(writing_height_mm=9,line_spacing_mm=13.5,naturalness=.8,
        line_start_variation=1,letter_variation=0,fit_lines_individually=True)
    original=layout_document(document,settings).pages[0]
    adjusted=layout_document(document,settings.model_copy(update={'line_end_variation':1})).pages[0]
    repeat=layout_document(document,settings.model_copy(update={'line_end_variation':1})).pages[0]
    starts=lambda rows:np.array([np.concatenate(l.strokes)[:,0].min() for l in rows])
    ends=lambda rows:np.array([np.concatenate(l.strokes)[:,0].max() for l in rows])
    np.testing.assert_allclose(starts(original),starts(adjusted),atol=1e-9)
    assert np.std(starts(adjusted))>.3
    assert (ends(adjusted)<ends(original)-.2).any()
    for a,b in zip(adjusted,repeat):
        for stroke_a,stroke_b in zip(a.strokes,b.strokes):np.testing.assert_array_equal(stroke_a,stroke_b)


def test_relaxed_layout_is_repeatable_with_varied_starts_and_optional_tidy_mode(document):
    document.lines *= 12
    settings = PageSettings(naturalness=.6)
    first = layout_document(document,settings).pages[0]
    repeat = layout_document(document,settings).pages[0]
    other = layout_document(document,settings.model_copy(update={"layout_seed":30})).pages[0]
    starts = [np.concatenate(line.strokes)[:,0].min() for line in first]
    assert np.std(starts) > .25
    assert not np.array_equal(first[0].strokes[0],other[0].strokes[0])
    for a,b in zip(first,repeat):
        for stroke_a,stroke_b in zip(a.strokes,b.strokes):
            np.testing.assert_array_equal(stroke_a,stroke_b)
    tidy = layout_document(document,settings.model_copy(update={"naturalness":0,"letter_variation":0})).pages[0]
    assert all(np.concatenate(line.strokes)[:,0].min() == settings.margin_mm for line in tidy)


def test_local_variation_changes_repeated_shapes_but_keeps_coincident_joins():
    shape = np.array([[0.,0.],[.5,-2.],[1.,-1.],[1.5,0.]])
    strokes = [shape+np.array([10.+i*3,20.]) for i in range(12)]
    strokes.append(strokes[4][-1:].copy())
    varied = _vary_letters(strokes,20.,"a"*24,.8,np.random.default_rng(29))
    np.testing.assert_array_equal(varied[4][-1],varied[-1][0])
    original = np.ptp(shape,axis=0)
    proportions = np.array([np.ptp(s,axis=0)/original for s in varied[:-1]])
    assert np.std(proportions[:,0]) > .01
    assert np.std(proportions[:,1]) > .01
    assert np.all((proportions > .65) & (proportions < 1.35))
    again = _vary_letters(strokes,20.,"a"*24,.8,np.random.default_rng(29))
    for a,b in zip(varied,again):np.testing.assert_array_equal(a,b)


def test_local_variation_preserves_baseline_order():
    baseline = np.column_stack((np.linspace(0,100,1001),np.zeros(1001)))
    mapped = _vary_letters([baseline],0.,"a"*80,1.,np.random.default_rng(4))[0]
    assert np.all(np.diff(mapped[:,0]) > 0)
    assert abs(mapped[-1,0]-mapped[0,0]-100) < 1e-6


def test_word_gap_variation_moves_groups_without_deforming_letters():
    shape = np.array([[0.,0.],[1.,-2.],[2.,0.]])
    strokes = [shape+np.array([x,20.]) for x in (10.,18.,27.)]
    varied = _vary_word_gaps(strokes,"one two three",1.,np.random.default_rng(19))
    assert not np.array_equal(varied[1],strokes[1])
    for a,b in zip(strokes,varied):
        np.testing.assert_allclose(np.diff(a,axis=0),np.diff(b,axis=0))
    assert all(b[:,0].min()>a[:,0].max() for a,b in zip(varied,varied[1:]))
    np.testing.assert_array_equal(strokes[0],shape+np.array([10.,20.]))


def test_word_gap_floor_opens_blank_intervals_without_changing_letter_geometry():
    shape = np.array([[0.,0.],[1.,-2.],[2.,0.]])
    strokes = [shape,shape+np.array([3.,0.])]
    opened = _vary_word_gaps(strokes,'one two',1,np.random.default_rng(19),2)
    assert opened[1][:,0].min()-opened[0][:,0].max() >= 2-1e-9
    for before,after in zip(strokes,opened):
        np.testing.assert_allclose(np.diff(before,axis=0),np.diff(after,axis=0))
    # Pen lifts inside a single word must not be expanded into word spaces.
    untouched = _vary_word_gaps(strokes,'letter',1,np.random.default_rng(19),2)
    for before,after in zip(strokes,untouched):np.testing.assert_array_equal(before,after)


def test_letter_width_and_height_controls_can_be_isolated():
    source = np.array([[0.,0.],[1.,-2.],[2.,-1.],[3.,0.],[5.,-2.],[7.,0.]])
    def vary(width,height):
        return _vary_letters([source],0.,"letter",1.,np.random.default_rng(12),width,height,0,0)[0]
    width_only,height_only = vary(1,0),vary(0,1)
    np.testing.assert_allclose(width_only[:,1],source[:,1])
    np.testing.assert_allclose(height_only[:,0],source[:,0],atol=1e-12)
    assert not np.allclose(width_only[:,0],source[:,0])
    assert not np.allclose(height_only[:,1],source[:,1])
    np.testing.assert_allclose(vary(0,0),source,atol=1e-12)
