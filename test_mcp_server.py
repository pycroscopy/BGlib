#!/usr/bin/env python3
"""
Simple test script to verify BGlib MCP server is working.
This script acts as an MCP client to test the server.
"""

import json
import subprocess
import sys
from typing import Dict, Any


def send_request(process: subprocess.Popen, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Send an MCP request to the server process."""
    request_id = 1
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {}
    }

    # Send request
    process.stdin.write(json.dumps(request).encode() + b'\n')
    process.stdin.flush()

    # Read response
    response_line = process.stdout.readline().decode().strip()
    if response_line:
        response = json.loads(response_line)
        return response
    return {}


def test_server():
    """Test the BGlib MCP server by starting it and making requests."""
    print("Starting BGlib MCP server...")

    # Start the server process
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "bglib_mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # We'll handle bytes
            cwd="/Users/rvv/Github/BGlib"
        )
    except FileNotFoundError:
        print("Error: Could not find bglib_mcp_server module. Make sure BGlib is installed with MCP support.")
        return False

    try:
        # Wait a bit for server to start
        import time
        time.sleep(1)

        # Test 1: List available tools
        print("\n1. Testing tool listing...")
        response = send_request(process, "tools/list")
        if "result" in response and "tools" in response["result"]:
            tools = [tool["name"] for tool in response["result"]["tools"]]
            print(f"✓ Available tools: {', '.join(tools)}")
        else:
            print("✗ Failed to list tools")
            print(f"Response: {response}")
            return False

        # Test 2: Call a simple tool (get_rotation_matrix)
        print("\n2. Testing tool call (get_rotation_matrix)...")
        response = send_request(process, "tools/call", {
            "name": "get_rotation_matrix",
            "arguments": {"theta": 0.0}
        })
        if "result" in response:
            print("✓ get_rotation_matrix call successful")
            print(f"Result: {response['result']}")
        else:
            print("✗ get_rotation_matrix call failed")
            print(f"Response: {response}")
            return False

        print("\n✓ All tests passed! BGlib MCP server is working correctly.")
        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)