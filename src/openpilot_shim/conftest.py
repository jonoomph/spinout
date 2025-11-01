import pytest


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture():
  """Override the repo-wide autouse fixture to keep tests dependency-free."""
  yield
