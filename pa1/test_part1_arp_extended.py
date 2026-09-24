"""Additional Part I tests, meant to run alongside test_part1_arp.py.

Covers the five hidden-test categories named in the handout (section 1.3):
multiple simultaneous unresolved next hops, FIFO release of queued
datagrams, replacement/refresh of learned mappings, malformed input, and
frames addressed to another host -- plus two bonus edge cases (independent
per-IP timeouts, and a single large tick() jump) motivated by the handout's
warning: "Avoid hard-coding addresses or assuming that only one ARP
resolution can be active."
"""

from csc458_pa1.network_interface import NetworkInterface
from csc458_pa1.protocols import (
    ARPMessage, ARP_REPLY, ARP_REQUEST, ETHERNET_BROADCAST,
    ETHERTYPE_ARP, ETHERTYPE_IPV4, EthernetFrame, IPv4Packet,
)

LOCAL_MAC = "02:00:00:00:00:10"
LOCAL_IP = "10.0.0.10"
ROUTER_MAC = "02:00:00:00:00:01"
ROUTER_IP = "10.0.0.1"

HOST_B_MAC = "02:00:00:00:00:02"
HOST_B_IP = "10.0.0.2"
HOST_C_MAC = "02:00:00:00:00:03"
HOST_C_IP = "10.0.0.3"
OTHER_MAC = "02:00:00:00:00:99"  # belongs to neither LOCAL nor broadcast


def datagram(src="192.0.2.10", dst="203.0.113.10", payload=b"hello", ttl=64):
    return IPv4Packet.build(src, dst, payload, protocol=17, ttl=ttl)


def arp_frame(src_mac, dst_mac, arp):
    return EthernetFrame(dst_mac, src_mac, ETHERTYPE_ARP, arp.to_bytes())


# ---------------------------------------------------------------------------
# Multiple simultaneous unresolved next hops
# ---------------------------------------------------------------------------

def test_multiple_simultaneous_unresolved_next_hops():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    iface.send_datagram(datagram(payload=b"to-router"), ROUTER_IP)
    iface.send_datagram(datagram(payload=b"to-hostb"), HOST_B_IP)

    # Two independent ARP requests, one per unresolved next hop. Order
    # between the two isn't asserted, just that both exist.
    first = iface.maybe_send()
    second = iface.maybe_send()
    assert first is not None and second is not None
    assert iface.maybe_send() is None

    requests = {}
    for frame in (first, second):
        assert frame.ethertype == ETHERTYPE_ARP and frame.dst == ETHERNET_BROADCAST
        arp = ARPMessage.parse(frame.payload)
        assert arp.opcode == ARP_REQUEST
        requests[arp.target_ip] = arp
    assert set(requests) == {ROUTER_IP, HOST_B_IP}

    # Resolving Host B must only release Host B's datagram; the router's
    # datagram must remain pending.
    reply_b = ARPMessage.reply(HOST_B_MAC, HOST_B_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(HOST_B_MAC, LOCAL_MAC, reply_b))

    frame = iface.maybe_send()
    assert frame is not None and frame.dst == HOST_B_MAC and frame.ethertype == ETHERTYPE_IPV4
    assert IPv4Packet.parse(frame.payload).payload == b"to-hostb"
    assert iface.maybe_send() is None

    # Now resolve the router; its datagram should be released too.
    reply_r = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply_r))

    frame = iface.maybe_send()
    assert frame is not None and frame.dst == ROUTER_MAC and frame.ethertype == ETHERTYPE_IPV4
    assert IPv4Packet.parse(frame.payload).payload == b"to-router"
    assert iface.maybe_send() is None


def test_independent_pending_timeout_per_next_hop():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    iface.send_datagram(datagram(payload=b"to-router"), ROUTER_IP)
    assert iface.maybe_send() is not None  # ARP request #1, started at t=0

    iface.tick(3_000)
    iface.send_datagram(datagram(payload=b"to-hostb"), HOST_B_IP)
    assert iface.maybe_send() is not None  # ARP request #2, started at t=3000

    # By t=5000: router's pending state (age 5000ms) expires and its
    # datagram is dropped; Host B's pending state (age 2000ms) must survive.
    iface.tick(2_000)

    reply_b = ARPMessage.reply(HOST_B_MAC, HOST_B_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(HOST_B_MAC, LOCAL_MAC, reply_b))
    frame = iface.maybe_send()
    assert frame is not None and frame.dst == HOST_B_MAC
    assert IPv4Packet.parse(frame.payload).payload == b"to-hostb"
    assert iface.maybe_send() is None

    # A late reply for the router has nothing left to release.
    reply_r = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply_r))
    assert iface.maybe_send() is None


def test_tick_handles_large_single_jump():
    # Bonus edge case: expiry must be computed correctly even when a lot of
    # time passes in one tick() call, not just when ticks happen to land
    # near the exact 5s/30s boundaries.
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    reply = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply))  # cached at t=0

    iface.send_datagram(datagram(payload=b"pending"), HOST_B_IP)
    assert iface.maybe_send() is not None  # ARP request pending at t=0

    iface.tick(60_000)  # jump well past both the 5s and 30s thresholds

    iface.send_datagram(datagram(payload=b"after-jump"), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_ARP  # cache was cleared

    # HOST_B_IP's queued datagram was dropped at the jump; a late reply
    # releases nothing.
    reply_b = ARPMessage.reply(HOST_B_MAC, HOST_B_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(HOST_B_MAC, LOCAL_MAC, reply_b))
    assert iface.maybe_send() is None


# ---------------------------------------------------------------------------
# FIFO release of several waiting datagrams
# ---------------------------------------------------------------------------

def test_fifo_release_of_several_waiting_datagrams():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    payloads = [b"first", b"second", b"third", b"fourth"]
    for p in payloads:
        iface.send_datagram(datagram(payload=p), ROUTER_IP)

    # Only the first send_datagram() call should have triggered an ARP
    # request; the rest are suppressed and queued behind it.
    request = iface.maybe_send()
    assert request is not None and request.ethertype == ETHERTYPE_ARP
    assert iface.maybe_send() is None

    reply = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply))

    released = []
    frame = iface.maybe_send()
    while frame is not None:
        assert frame.ethertype == ETHERTYPE_IPV4
        released.append(IPv4Packet.parse(frame.payload).payload)
        frame = iface.maybe_send()

    assert released == payloads


# ---------------------------------------------------------------------------
# Replacement / refresh of learned mappings
# ---------------------------------------------------------------------------

def test_arp_mapping_replacement_updates_mac():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    reply1 = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply1))

    iface.send_datagram(datagram(payload=b"before"), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.dst == ROUTER_MAC

    # The router's NIC changes; it ARPs for us using a new MAC. This should
    # both learn/replace the cached mapping AND (since it targets our IP)
    # queue a reply.
    new_router_mac = "02:00:00:00:00:aa"
    announce = ARPMessage.request(new_router_mac, ROUTER_IP, LOCAL_IP)
    iface.recv_frame(arp_frame(new_router_mac, ETHERNET_BROADCAST, announce))

    reply_frame = iface.maybe_send()
    assert reply_frame is not None and reply_frame.ethertype == ETHERTYPE_ARP
    assert reply_frame.dst == new_router_mac
    assert ARPMessage.parse(reply_frame.payload).opcode == ARP_REPLY
    assert iface.maybe_send() is None

    # Subsequent datagrams to the router must use the new MAC.
    iface.send_datagram(datagram(payload=b"after"), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.dst == new_router_mac and frame.ethertype == ETHERTYPE_IPV4
    assert IPv4Packet.parse(frame.payload).payload == b"after"


def test_arp_mapping_refresh_resets_expiry_timer():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    reply = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, LOCAL_MAC, LOCAL_IP)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply))

    iface.tick(29_999)
    iface.recv_frame(arp_frame(ROUTER_MAC, LOCAL_MAC, reply))  # refresh

    # Without a timer reset this would be 29,999 + 29,999 ms old and long
    # expired. With a reset, it's only 29,999 ms since the refresh.
    iface.tick(29_999)
    iface.send_datagram(datagram(payload=b"still-cached"), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_IPV4

    # 30,000ms after the refresh, it finally expires.
    iface.tick(1)
    iface.send_datagram(datagram(payload=b"expired-now"), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_ARP


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_malformed_arp_payload_is_ignored():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    garbage = EthernetFrame(LOCAL_MAC, ROUTER_MAC, ETHERTYPE_ARP, b"\x01\x02\x03")
    assert iface.recv_frame(garbage) is None
    assert iface.maybe_send() is None  # no reply was queued

    # Nothing should have been learned from it either.
    iface.send_datagram(datagram(), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_ARP


def test_malformed_ipv4_payload_is_ignored():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    truncated = EthernetFrame(LOCAL_MAC, ROUTER_MAC, ETHERTYPE_IPV4, b"\x45\x00\x00")
    assert iface.recv_frame(truncated) is None
    assert iface.maybe_send() is None


def test_arp_with_unsupported_opcode_is_ignored():
    # ARPMessage.parse() rejects opcodes other than ARP_REQUEST/ARP_REPLY.
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    weird = ARPMessage(1, 0x0800, 6, 4, 99, ROUTER_MAC, ROUTER_IP,
                        "00:00:00:00:00:00", LOCAL_IP)
    frame = EthernetFrame(LOCAL_MAC, ROUTER_MAC, ETHERTYPE_ARP, weird.to_bytes())
    assert iface.recv_frame(frame) is None
    assert iface.maybe_send() is None


def test_unknown_ethertype_is_ignored():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    frame = EthernetFrame(LOCAL_MAC, ROUTER_MAC, 0x86DD, b"not-ipv6-really")
    assert iface.recv_frame(frame) is None
    assert iface.maybe_send() is None


# ---------------------------------------------------------------------------
# Frames addressed to another host
# ---------------------------------------------------------------------------

def test_arp_request_unicast_to_another_host_is_discarded():
    # Ethernet-level filtering (dst is neither us nor broadcast) must win
    # even though the ARP payload's target_ip happens to be us.
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    request = ARPMessage.request(HOST_B_MAC, HOST_B_IP, LOCAL_IP)
    not_for_us = EthernetFrame(OTHER_MAC, HOST_B_MAC, ETHERTYPE_ARP, request.to_bytes())
    assert iface.recv_frame(not_for_us) is None
    assert iface.maybe_send() is None  # must not have replied

    # Host B's mapping must not have been learned from the discarded frame.
    iface.send_datagram(datagram(), HOST_B_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_ARP


def test_arp_reply_unicast_to_another_host_is_discarded():
    # Same idea as above, but with an ARP REPLY instead of a REQUEST -- the
    # discard rule is about the Ethernet destination, not the ARP opcode.
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    reply = ARPMessage.reply(ROUTER_MAC, ROUTER_IP, HOST_C_MAC, HOST_C_IP)
    not_for_us = EthernetFrame(HOST_C_MAC, ROUTER_MAC, ETHERTYPE_ARP, reply.to_bytes())
    assert iface.recv_frame(not_for_us) is None

    # The router's mapping must not have been learned from the discarded
    # reply either.
    iface.send_datagram(datagram(), ROUTER_IP)
    frame = iface.maybe_send()
    assert frame is not None and frame.ethertype == ETHERTYPE_ARP


def test_arp_broadcast_request_not_targeting_us_is_learned_but_not_replied():
    iface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    request = ARPMessage.request(HOST_B_MAC, HOST_B_IP, HOST_C_IP)
    frame = EthernetFrame(ETHERNET_BROADCAST, HOST_B_MAC, ETHERTYPE_ARP, request.to_bytes())
    assert iface.recv_frame(frame) is None
    assert iface.maybe_send() is None  # not the target: no reply

    iface.send_datagram(datagram(), HOST_B_IP)
    out = iface.maybe_send()
    assert out is not None and out.ethertype == ETHERTYPE_IPV4 and out.dst == HOST_B_MAC
