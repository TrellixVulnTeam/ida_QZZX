
def pytest_addoption(parser):
    parser.addoption("--validation-path", action="store")
    parser.addoption("--batch-size", type=int, default=512, action="store")


def pytest_generate_tests(metafunc):
    validation_path = metafunc.config.option.validation_path
    batch_size = metafunc.config.option.batch_size

    if 'validation_path' in metafunc.fixturenames \
            and validation_path is not None:
        metafunc.parametrize("validation_path", [validation_path])

    if 'batch_size' in metafunc.fixturenames \
            and batch_size is not None:
        metafunc.parametrize("batch_size", [batch_size])
