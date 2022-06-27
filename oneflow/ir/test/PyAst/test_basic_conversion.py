from onefow.ir.decorator import lr_def

@lr_def
def _test_bare_kernel():
    pass

@lr_def
def _test_kernel_with_variable():
    mock_res = 1

@lr_def
def _test_kernel_with_return():
    return 1

@lr_def
def _test_kernel_with_arith():
    mock_a = 1
    mock_b = 1
    mock_res = mock_a + mock_b - ( mock_a * mock_b) / 2
    return mock_res

@lr_def
def _test_kernel_with_arith():
    mock_a = 1
    mock_res = 0
    mock_res += mock_a
    mock_res -= mock_a
    mock_res *= mock_a
    mock_res /= mock_a
    return mock_res

@lr_def
def _test_kernel_with_params(mock_a, mock_b):
    mock_res = mock_a + mock_b - ( mock_a * mock_b) / 2
    return mock_res

@lr_def
def _test_kernel_with_if(mock_bool):
    if mock_bool:
        return 1
    else:
        return 0

@lr_def
def _test_kernel_with_tri(mock_bool):
    mock_res = mock_bool if 1 else 0
    return mock_res

@lr_def
def _test_kernel_with_for():
    mock_a = 1
    for i in [1, 2, 3, 4]:
        mock_a += i

@lr_def
def _test_kernel_with_range():
    mock_a = 1
    for i in range(3):
        mock_a += i


def test_bare_kernel():
    _test_bare_kernel()

def test_kernel_with_variable():
    _test_kernel_with_variable()

def test_kernel_with_return():
    _test_kernel_with_return()

def test_kernel_with_arith():
    _test_kernel_with_arith()


def test_kernel_with_arith():
    _test_kernel_with_arith()

def test_kernel_with_params():
    _test_kernel_with_params(10, -1)

def test_kernel_with_if():
    _test_kernel_with_if(False)

def test_kernel_with_tri():
    _test_kernel_with_tri(True)

def test_kernel_with_for():
    _test_kernel_with_for()

def test_kernel_with_range():
    _test_kernel_with_range()
