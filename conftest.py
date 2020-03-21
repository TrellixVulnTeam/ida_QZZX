
def pytest_addoption(parser):
    parser.addoption("--validation-url", action="store")
    parser.addoption("--batch-size", type=int, default=512, action="store")


def pytest_generate_tests(metafunc):
    validation_url = metafunc.config.option.validation_url
    batch_size = metafunc.config.option.batch_size

    if 'validation_url' in metafunc.fixturenames \
            and validation_url is not None:
        metafunc.parametrize("validation_url", [validation_url])

    if 'batch_size' in metafunc.fixturenames \
            and batch_size is not None:
        metafunc.parametrize("batch_size", [batch_size])
