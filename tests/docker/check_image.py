"""Teste de integração Docker: imagem limpa, handlers reais e SIGTERM sem rede."""
import base64
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "contraponto-bot:local"


def docker(*args):
    result = subprocess.run(
        ["docker", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60,
    )
    return (result.stdout + result.stderr).strip() if args[0] == "logs" else result.stdout.strip()


def main():
    # Nenhum .env do host é passado ou montado neste teste.
    docker("run", "--rm", "--network", "none", IMAGE, "python", "-c", (
        "from pathlib import Path; import os, bot, claims, search, verifier; "
        "assert os.getuid() != 0; "
        "assert not list(Path('/app').rglob('.env*')); "
        "assert not list(Path('/app').rglob('*.joblib')); "
        "assert all(not (Path('/app') / name).exists() "
        "for name in ['.git','.venv','data','models','reports','docs','requirements','scripts','tests']); "
        "print('Conteudo da imagem e imports: OK')"
    ))
    model_probe = ROOT / "tests/docker/model_probe.py"
    probe_mount = f"type=bind,source={model_probe},target=/model_probe.py,readonly"
    # Gera o artefato na própria imagem para garantir as mesmas versões de ML.
    fixture = json.loads(docker(
        "run", "--rm", "--network", "none", "--mount", probe_mount,
        IMAGE, "python", "/model_probe.py", "fixture",
    ))
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".docker-test-model-") as folder:
        models = Path(folder)
        models.chmod(0o755)
        language_dir = models / "language_risk"
        language_dir.mkdir()
        # Sem o bind somente leitura, o usuário do contêiner poderia escrever aqui.
        language_dir.chmod(0o777)
        artifact = base64.b64decode(fixture["artifact"])
        model = language_dir / "baseline_tfidf_logreg.joblib"
        model.write_bytes(artifact)
        model.chmod(0o644)
        print(docker(
            "run", "--rm", "--network", "none", "--mount", probe_mount,
            "--mount", f"type=bind,source={models},target=/app/models,readonly",
            "--env", "LANGUAGE_RISK_MODEL_PATH=/app/models/language_risk/baseline_tfidf_logreg.joblib",
            IMAGE, "python", "/model_probe.py", "verify", str(fixture["expected"]),
        ))
        assert model.read_bytes() == artifact, "A inferência alterou o artefato"
    name = "contraponto-smoke-" + uuid.uuid4().hex[:10]
    probe = ROOT / "tests/docker/polling_probe.py"
    try:
        docker(
            "run", "--detach", "--name", name, "--init", "--network", "none",
            "--mount", f"type=bind,source={probe},target=/polling_probe.py,readonly",
            IMAGE, "python", "/polling_probe.py",
        )
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            logs = docker("logs", name)
            if "SMOKE_READY:" in logs:
                break
            if docker("inspect", "--format", "{{.State.Running}}", name) != "true":
                raise AssertionError("Polling terminou antes de processar mensagens:\n" + logs)
            time.sleep(0.5)
        else:
            raise AssertionError("Timeout aguardando handlers do bot:\n" + docker("logs", name))
        docker("stop", "--time", "15", name)
        assert docker("inspect", "--format", "{{.State.ExitCode}}", name) == "0", "SIGTERM não encerrou com sucesso"
        logs = docker("logs", name)
        assert "SMOKE_STOPPED:" in logs, "Polling não encerrou normalmente"
        assert "ci-fake-token" not in logs, "Token apareceu nos logs"
        print("Docker: /start, fallback sem OpenAI e SIGTERM passaram.")
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)


if __name__ == "__main__":
    main()
