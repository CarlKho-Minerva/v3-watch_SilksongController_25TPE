#!/usr/bin/env python3
"""
Quick Connection Test Script

Tests the improved connection verification without starting a full recording.
Useful for debugging connection issues between watch and computer.

Usage:
    python test_connection.py
"""

import socket
import json
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network_utils

# ANSI Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def test_connection():
    """Test connection to watch"""
    print(f"\n{Colors.CYAN}╔══════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.CYAN}║     WATCH CONNECTION TEST - Quick Diagnostic     ║{Colors.RESET}")
    print(f"{Colors.CYAN}╚══════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Auto-detect IP
    print(f"{Colors.CYAN}🔍 Auto-detecting IP address...{Colors.RESET}")
    network_utils.update_config_ip()
    config = load_config()

    listen_ip = config["network"]["listen_ip"]
    listen_port = config["network"]["listen_port"]

    print(f"{Colors.GREEN}✓ Will listen on {listen_ip}:{listen_port}{Colors.RESET}\n")

    # Set up socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((listen_ip, listen_port))
        sock.setblocking(False)
        print(f"{Colors.GREEN}✓ Socket bound successfully{Colors.RESET}")
    except OSError as e:
        print(f"{Colors.RED}❌ ERROR: Could not bind to {listen_ip}:{listen_port}{Colors.RESET}")
        print(f"{Colors.YELLOW}Make sure no other listener is running!{Colors.RESET}")
        return False

    print(f"\n{Colors.YELLOW}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}WAITING FOR WATCH DATA{Colors.RESET}")
    print(f"{Colors.YELLOW}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}Please ensure:{Colors.RESET}")
    print(f"  {Colors.BLUE}1. Watch app is open{Colors.RESET}")
    print(f"  {Colors.BLUE}2. Streaming toggle is ON{Colors.RESET}")
    print(f"  {Colors.BLUE}3. Watch shows 'Connected!' in green{Colors.RESET}")
    print(f"  {Colors.BLUE}4. Both devices on same WiFi{Colors.RESET}\n")

    # Wait for packets
    timeout = 15
    start_time = time.time()
    last_update = 0
    packet_count = 0
    sensor_types = set()

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed

        # Update status
        if time.time() - last_update > 0.5:
            status = f"⏳ Waiting... {remaining:.1f}s remaining"
            if packet_count > 0:
                status += f" | {Colors.GREEN}{packet_count} packets{Colors.RESET}"
            print(f"\r{Colors.YELLOW}{status}{Colors.RESET}", end="", flush=True)
            last_update = time.time()

        try:
            data, addr = sock.recvfrom(4096)
            parsed = json.loads(data.decode())

            if "sensor" in parsed:
                packet_count += 1
                sensor_type = parsed["sensor"]
                sensor_types.add(sensor_type)

                # Success after 5 packets
                if packet_count >= 5:
                    print(f"\n\n{Colors.GREEN}{'='*70}{Colors.RESET}")
                    print(f"{Colors.GREEN}✅ CONNECTION SUCCESSFUL!{Colors.RESET}")
                    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")

                    print(f"{Colors.BOLD}Connection Details:{Colors.RESET}")
                    print(f"  {Colors.CYAN}Packets received:{Colors.RESET} {packet_count}")
                    print(f"  {Colors.CYAN}Watch address:{Colors.RESET} {addr[0]}:{addr[1]}")
                    print(f"  {Colors.CYAN}Sensor types:{Colors.RESET} {', '.join(sorted(sensor_types))}")
                    print(f"  {Colors.CYAN}Time to connect:{Colors.RESET} {elapsed:.1f}s")

                    # Show sample packet
                    print(f"\n{Colors.BOLD}Sample Packet:{Colors.RESET}")
                    print(f"  {Colors.BLUE}{json.dumps(parsed, indent=2)}{Colors.RESET}")

                    print(f"\n{Colors.GREEN}✨ Your watch is connected and streaming properly!{Colors.RESET}")
                    print(f"{Colors.GREEN}✓ Ready for data collection{Colors.RESET}\n")

                    sock.close()
                    return True

        except BlockingIOError:
            time.sleep(0.05)
        except json.JSONDecodeError as e:
            print(f"\n{Colors.YELLOW}⚠️  Malformed packet: {e}{Colors.RESET}")
        except KeyError:
            pass

    # Timeout
    print(f"\n\n{Colors.RED}{'='*70}{Colors.RESET}")
    print(f"{Colors.RED}❌ CONNECTION TIMEOUT{Colors.RESET}")
    print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")

    if packet_count > 0:
        print(f"{Colors.YELLOW}⚠️  Received {packet_count} packets - connection is slow/intermittent{Colors.RESET}")
        print(f"  {Colors.CYAN}Sensor types seen:{Colors.RESET} {', '.join(sorted(sensor_types))}")
        print(f"\n{Colors.YELLOW}This might work but data rate is low.{Colors.RESET}")
        print(f"{Colors.YELLOW}Check WiFi signal strength on watch.{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}No packets received from watch!{Colors.RESET}\n")

        print(f"{Colors.BOLD}Troubleshooting Steps:{Colors.RESET}")
        print(f"  {Colors.YELLOW}1. Watch App:{Colors.RESET}")
        print(f"     - Open the Silksong Controller app")
        print(f"     - Toggle streaming switch ON")
        print(f"     - Look for green 'Connected!' message")
        print(f"  {Colors.YELLOW}2. Network:{Colors.RESET}")
        print(f"     - Ensure both devices on same WiFi")
        print(f"     - Check firewall isn't blocking port {listen_port}")
        print(f"  {Colors.YELLOW}3. IP Address:{Colors.RESET}")
        print(f"     - Computer IP: {listen_ip}")
        print(f"     - Verify watch app has same IP")
        print(f"     - Try manual IP entry on watch")
        print(f"  {Colors.YELLOW}4. Restart:{Colors.RESET}")
        print(f"     - Close and reopen watch app")
        print(f"     - Toggle streaming OFF then ON")
        print(f"     - Run this test again\n")

    sock.close()
    return False


if __name__ == "__main__":
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Test interrupted{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ ERROR: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
