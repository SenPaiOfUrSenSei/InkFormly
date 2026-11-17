import time

from fastapi.testclient import TestClient

from inkform.server import create_app
from inkform.schema import RenderSettings


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["state"] in ("ready", "error"):
            return result
        time.sleep(0.02)
    raise AssertionError("Background job did not complete")


def test_generate_recolor_and_download_reuses_trajectories(document, tmp_path):
    class FakeGenerator:
        calls = 0

        def generate(self, settings, progress):
            self.calls += 1
            progress(0, 1, settings.text)
            return document

        def rewrite_line(self, original, index):
            updated = original.model_copy(deep=True)
            updated.lines[index].seed += 1
            return updated

    generator = FakeGenerator()
    with TestClient(create_app(tmp_path, lambda: generator)) as client:
        response = client.post("/api/generate", json={"text": "A test line"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert wait_for_job(client, job_id)["state"] == "ready"
        original = client.get(f"/api/jobs/{job_id}/files/document.json").json()
        review = client.get(f"/api/jobs/{job_id}/review").json()
        assert review["lines"][0]["text"] == "A test line"
        assert review["lines"][0]["readback"]["status"] == "unchecked"
        assert client.get(f"/api/jobs/{job_id}/files/line-001.png").headers["content-type"] == "image/png"
        assert client.get(f"/api/jobs/{job_id}/files/preview-001.png").headers["content-type"] == "image/png"
        preset = client.get('/api/variation-presets').json()['lively']
        updated = client.post(f"/api/jobs/{job_id}/render", json={
            "ink": {**preset['ink'],"color": "#111111"},
            "page": {**preset['page'],"layout_seed":101}})
        updated_id = updated.json()["id"]
        assert wait_for_job(client, updated_id)["state"] == "ready"
        assert client.get(f"/api/jobs/{updated_id}/files/document.json").json() == original
        assert generator.calls == 1
        import json
        saved = json.loads((tmp_path/updated_id/'settings.json').read_text())
        assert saved['page']['layout_seed'] == 101
        assert saved['page']['word_spacing_variation'] == preset['page']['word_spacing_variation']
        export_id = client.post(f"/api/jobs/{updated_id}/export", json={}).json()["id"]
        exported = wait_for_job(client, export_id)
        assert exported["state"] == "ready"
        assert exported["export_ready"]
        assert client.get(f"/api/jobs/{export_id}/files/inkform.pdf").content.startswith(b"%PDF")
        assert client.get(f"/api/jobs/{export_id}/files/pages.zip").content.startswith(b"PK")
        assert generator.calls == 1
        assert client.get(f"/api/jobs/{updated_id}/files/reference.json").json()["text"] == "A test line"
        assert client.get(f"/api/jobs/{updated_id}/files/settings.json").status_code == 404
        assert client.post("/api/generate", json={"text": " "}).status_code == 422
        assert client.post("/api/generate", json={"text": "x", "ink": {"color": "red"}}).status_code == 422
        assert client.post("/api/jobs/unknown/render", json={}).status_code == 404
        assert client.post(f"/api/jobs/{job_id}/lines/99/rewrite", json={}).status_code == 422
        rewritten_id = client.post(f"/api/jobs/{job_id}/lines/0/rewrite", json={}).json()["id"]
        assert wait_for_job(client, rewritten_id)["state"] == "ready"
        rewritten = client.get(f"/api/jobs/{rewritten_id}/files/document.json").json()
        assert rewritten["source_text"] == original["source_text"]
        assert rewritten["lines"][0]["seed"] == original["lines"][0]["seed"] + 1


def test_background_errors_are_reported(tmp_path):
    class Broken:
        def generate(self, settings, progress):
            raise RuntimeError("The model could not finish this line.")

    with TestClient(create_app(tmp_path, Broken)) as client:
        job_id = client.post("/api/generate", json={"text": "hello"}).json()["id"]
        job = wait_for_job(client, job_id)
        assert job["state"] == "error"
        assert "could not finish" in job["error"]
        assert client.post(f"/api/jobs/{job_id}/export", json={}).status_code == 409


def test_inconclusive_t_words_remain_visible_after_matching_full_line_readback(tmp_path,document):
    from inkform.recognition import compare_readback
    document.quality_checks=True
    document.lines[0].readback=compare_readback(document.lines[0].text,document.lines[0].text)
    document.lines[0].t_formation=dict(method='whole-line-word-clarity-v1',selected_words=None)
    class Generator:
        def generate(self,settings,progress):return document
    with TestClient(create_app(tmp_path,Generator)) as client:
        job_id=client.post('/api/generate',json={'text':'A test line'}).json()['id']
        job=wait_for_job(client,job_id)
        assert job['state']=='ready' and job['readback_matches']==1
        assert job['review_lines']==[1]
        review=client.get(f'/api/jobs/{job_id}/review').json()['lines'][0]
        assert review['readback']['status']=='match'
        assert review['word_clarity']['status']=='review'


def test_variation_presets_are_valid_render_settings_without_synthesis(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get('/api/variation-presets')
        assert response.status_code == 200
        presets = response.json()
        assert set(presets) == {'readable','light','balanced','lively','loose'}
        for preset in presets.values():
            settings = RenderSettings(page=preset['page'],ink=preset['ink'])
            assert settings.page.word_spacing_variation > 0
            assert settings.ink.density_variation > 0


def test_withdrawn_r_notice_survives_legacy_document_render_without_changing_strokes(document,tmp_path):
    document.cursive_r=True
    document.lines[0].r_formation=dict(applied=True)
    document.warnings=[]  # Old saves predate the withdrawal notice.
    snapshot=document.model_dump(mode='json')

    class LegacyGenerator:
        def generate(self,settings,progress):
            return document

    with TestClient(create_app(tmp_path,lambda:LegacyGenerator())) as client:
        original_id=client.post('/api/generate',json={'text':document.source_text}).json()['id']
        original=wait_for_job(client,original_id)
        assert any('withdrawn r-style edits' in w for w in original['warnings'])
        assert client.get(f'/api/jobs/{original_id}/files/document.json').json()==snapshot
        rendered_id=client.post(f'/api/jobs/{original_id}/render',json={}).json()['id']
        rendered=wait_for_job(client,rendered_id)
        assert any('withdrawn r-style edits' in w for w in rendered['warnings'])
        assert client.get(f'/api/jobs/{rendered_id}/files/document.json').json()==snapshot
