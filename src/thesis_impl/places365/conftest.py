
def pytest_addoption(parser):
    parser.addoption("--validation-path", action="store")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.validation_path
    if 'validation_path' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("validation_path", [option_value])
