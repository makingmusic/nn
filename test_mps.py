# this program checks if the  M1 M2 M3 chip on a macbook is actually used instead of the CPU when I am using PyTorch

import time, torch, platform

def print_test_status(test_name, passed):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{test_name}: {status}")

print("PyTorch:", torch.__version__, "| macOS:", platform.mac_ver()[0])

# Test 1: Check MPS availability
print("\nTest 1: MPS Availability")
mps_built = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()
print("MPS built into wheel?  ", mps_built)
print("MPS device available? ", mps_available)
print_test_status("Test 1", mps_built and mps_available)

# Test 2: Basic MPS operation
print("\nTest 2: Basic MPS Operation")
try:
    device = torch.device("mps")
    x = torch.randn(8_000, 8_000, device=device)
    y = torch.mm(x, x)
    test2_passed = y.device.type == "mps"
    print(f"Result device: {y.device}")
    print_test_status("Test 2", test2_passed)
except Exception as e:
    print(f"Error in Test 2: {str(e)}")
    print_test_status("Test 2", False)

# Test 3: Performance comparison
print("\nTest 3: Performance Comparison")
def matmul_on(dev):
    try:
        # Using much larger matrices to better demonstrate GPU advantage
        large_number = 32000
        a = torch.randn(large_number, large_number, device=dev)
        b = torch.randn(large_number, large_number, device=dev)
        torch.cuda.synchronize() if dev.type == "cuda" else torch.mps.synchronize() if dev.type=="mps" else None
        t0 = time.perf_counter()
        c = a @ b
        torch.mps.synchronize() if dev.type=="mps" else None
        return time.perf_counter() - t0
    except Exception as e:
        print(f"Error in matmul_on: {str(e)}")
        return float('inf')

try:
    cpu_time = matmul_on(torch.device("cpu"))
    mps_time = matmul_on(torch.device("mps"))
    print(f"CPU time: {cpu_time:.2f}s")
    print(f"MPS time: {mps_time:.2f}s")
    # Consider test passed if MPS is faster than CPU
    test3_passed = mps_time < cpu_time
    print_test_status("Test 3", test3_passed)
except Exception as e:
    print(f"Error in Test 3: {str(e)}")
    print_test_status("Test 3", False)