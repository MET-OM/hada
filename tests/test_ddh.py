from click.testing import CliRunner
from ddh.cli import ddh

def test_run(tmpdir):
    ru = CliRunner()
    with ru.isolated_filesystem(temp_dir=tmpdir) as td:
        print(tmpdir)
        r = ru.invoke(ddh, ['--output', tmpdir])
        assert r.exit_code == 0
