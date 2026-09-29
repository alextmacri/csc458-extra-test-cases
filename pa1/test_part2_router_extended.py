"""Additional Part II tests for routing state and the Router/ARP boundary."""

from dataclasses import dataclass

import pytest

from csc458_pa1.network_interface import NetworkInterface
from csc458_pa1.prefix_tools import (
    canonical_prefix,
    longest_prefix_match,
    prefix_contains,
    prefix_length,
)
from csc458_pa1.protocols import (
    ARPMessage,
    ARP_REQUEST,
    ETHERNET_BROADCAST,
    ETHERTYPE_ARP,
    ETHERTYPE_IPV4,
    EthernetFrame,
    IPv4Packet,
)
from csc458_pa1.router import Router


@dataclass
class RecordingInterface:
    sent: list

    def __init__(self):
        self.sent = []

    def send_datagram(self, datagram, next_hop_ip):
        self.sent.append((datagram, next_hop_ip))


def pkt(dst, ttl=8, payload=b"x"):
    return IPv4Packet.build("192.0.2.9", dst, payload, protocol=17, ttl=ttl)


def test_route_with_same_canonical_prefix_replaces_previous_entry():
    ifaces = [RecordingInterface(), RecordingInterface()]
    r = Router(ifaces)
    r.add_route("10.8.9.10", 16, "192.0.2.1", 0)
    r.add_route("10.8.0.1", 16, None, 1)

    assert len(r.routes) == 1
    assert r.routes[0].prefix == "10.8.0.0/16"
    assert r.routes[0].next_hop is None
    assert r.routes[0].interface_num == 1

    r.receive_datagram(0, pkt("10.8.7.7"))
    r.route()
    assert ifaces[0].sent == []
    assert ifaces[1].sent[0][1] == "10.8.7.7"


def test_router_forwards_queued_datagrams_in_fifo_order_without_mutating_inputs():
    interface = RecordingInterface()
    r = Router([RecordingInterface(), interface])
    r.add_route("10.9.0.0", 16, None, 1)
    packets = [pkt("10.9.0.1", ttl=5, payload=b"first"),
               pkt("10.9.0.2", ttl=6, payload=b"second")]

    for packet in packets:
        r.receive_datagram(0, packet)
    r.route()

    assert [sent[0].payload for sent in interface.sent] == [b"first", b"second"]
    assert [sent[0].ttl for sent in interface.sent] == [4, 5]
    assert [packet.ttl for packet in packets] == [5, 6]


def test_router_with_network_interface_emits_one_arp_request_and_releases_queue():
    local_mac = "02:00:00:00:00:10"
    local_ip = "10.0.0.10"
    next_hop_ip = "10.0.0.1"
    next_hop_mac = "02:00:00:00:00:01"
    interface = NetworkInterface(local_mac, local_ip)
    r = Router([interface])
    r.add_route("203.0.113.0", 24, next_hop_ip, 0)

    first = pkt("203.0.113.7", payload=b"first")
    second = pkt("203.0.113.8", payload=b"second")
    r.receive_datagram(0, first)
    r.receive_datagram(0, second)
    r.route()

    request = interface.maybe_send()
    assert request is not None
    assert (request.ethertype, request.dst) == (ETHERTYPE_ARP, ETHERNET_BROADCAST)
    arp = ARPMessage.parse(request.payload)
    assert (arp.opcode, arp.target_ip) == (ARP_REQUEST, next_hop_ip)
    assert interface.maybe_send() is None

    reply = ARPMessage.reply(next_hop_mac, next_hop_ip, local_mac, local_ip)
    interface.recv_frame(EthernetFrame(local_mac, next_hop_mac,
                                       ETHERTYPE_ARP, reply.to_bytes()))

    forwarded = []
    frame = interface.maybe_send()
    while frame is not None:
        assert frame.ethertype == ETHERTYPE_IPV4
        forwarded.append((frame.dst, IPv4Packet.parse(frame.payload)))
        frame = interface.maybe_send()
    assert [(dst, packet.payload) for dst, packet in forwarded] == [
        (next_hop_mac, b"first"),
        (next_hop_mac, b"second"),
    ]


def test_prefix_helpers_support_both_prefix_length_boundaries():
    assert canonical_prefix("203.0.113.99/32") == "203.0.113.99/32"
    assert canonical_prefix("203.0.113.99/0") == "0.0.0.0/0"
    assert prefix_length("203.0.113.99/32") == 32
    assert prefix_length("203.0.113.99/0") == 0
    assert prefix_contains("203.0.113.99/32", "203.0.113.99")
    assert not prefix_contains("203.0.113.99/32", "203.0.113.98")


def test_longest_prefix_match_canonicalizes_the_selected_host_bits():
    prefixes = ["0.0.0.0/0", "192.0.2.99/32", "192.0.2.0/24"]

    assert longest_prefix_match(prefixes, "192.0.2.99") == "192.0.2.99/32"
    assert longest_prefix_match(prefixes, "192.0.2.100") == "192.0.2.0/24"
    assert longest_prefix_match([], "192.0.2.100") is None


@pytest.mark.parametrize(
    "cidr",
    ["192.0.2.1/33", "192.0.2.1/-1", "192.0.2.1", "2001:db8::/32"],
)
def test_prefix_helpers_reject_non_ipv4_or_invalid_prefixes(cidr):
    with pytest.raises(ValueError):
        canonical_prefix(cidr)


def test_prefix_contains_rejects_an_ipv6_destination():
    with pytest.raises(ValueError):
        prefix_contains("192.0.2.0/24", "2001:db8::1")


def test_router_rejects_invalid_route_state_before_installing_it():
    interface = RecordingInterface()
    router = Router([interface])

    with pytest.raises(IndexError):
        router.add_route("10.0.0.0", 24, None, 1)
    with pytest.raises(ValueError):
        router.add_route("10.0.0.0", 33, None, 0)
    with pytest.raises(ValueError):
        router.add_route("10.0.0.0", 24, "2001:db8::1", 0)

    assert router.routes == ()


def test_router_handles_mixed_fifo_queue_and_can_route_again_after_draining():
    ingress = RecordingInterface()
    egress = RecordingInterface()
    router = Router([ingress, egress])
    router.add_route("10.0.0.0", 8, None, 1)

    first = pkt("198.51.100.1", ttl=5, payload=b"no route")
    second = pkt("10.1.2.3", ttl=3, payload=b"forward")
    third = pkt("10.4.5.6", ttl=1, payload=b"expired")
    fourth = pkt("10.7.8.9", ttl=2, payload=b"later")
    for datagram in (first, second, third, fourth):
        router.receive_datagram(0, datagram)

    router.route()

    assert [(datagram.payload, next_hop, datagram.ttl)
            for datagram, next_hop in egress.sent] == [
        (b"forward", "10.1.2.3", 2),
        (b"later", "10.7.8.9", 1),
    ]
    assert first.ttl == 5
    assert second.ttl == 3
    assert third.ttl == 1
    assert fourth.ttl == 2

    router.receive_datagram(0, pkt("10.9.9.9", ttl=4, payload=b"again"))
    router.route()
    assert egress.sent[-1][0].payload == b"again"


def test_router_rejects_invalid_ingress_interface():
    router = Router([RecordingInterface()])

    with pytest.raises(IndexError):
        router.receive_datagram(-1, pkt("10.0.0.1"))
    with pytest.raises(IndexError):
        router.receive_datagram(1, pkt("10.0.0.1"))
