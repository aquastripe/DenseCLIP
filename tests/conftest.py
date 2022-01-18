def pytest_addoption(parser):
    parser.addoption(
        '--device', type=str, help='device to store tensors'
    )
