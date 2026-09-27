"""Additional, deliberately verbose tests for NetworkInterface ARP behavior.

These tests focus on state interactions that are easy to get almost right:
separate unresolved next hops, FIFO release, cache replacement/refresh, and
ignoring malformed or incorrectly addressed Ethernet frames.
"""

import pytest

from csc458_pa1.network_interface import NetworkInterface
from csc458_pa1.protocols import (
    ARPMessage,
    ARP_REPLY,
    ARP_REQUEST,
    ETHERNET_BROADCAST,
    ETHERTYPE_ARP,
    ETHERTYPE_IPV4,
    EthernetFrame,
    IPv4Packet,
)


LOCAL_MAC = "02:00:00:00:00:10"
LOCAL_IP = "10.0.0.10"
NEXT_HOP_A = "10.0.0.1"
NEXT_HOP_B = "10.0.0.2"
NEXT_HOP_A_MAC_1 = "02:00:00:00:00:01"
NEXT_HOP_A_MAC_2 = "02:00:00:00:00:11"
NEXT_HOP_B_MAC = "02:00:00:00:00:02"
OTHER_HOST_MAC = "02:00:00:00:00:99"


def datagram(payload=b"payload", destination="203.0.113.10"):
    return IPv4Packet.build(
        "192.0.2.10", destination, payload, protocol=17, ttl=64
    )


def arp_frame(source_mac, destination_mac, message):
    return EthernetFrame(
        destination_mac, source_mac, ETHERTYPE_ARP, message.to_bytes()
    )


def receive_arp_reply(interface, sender_mac, sender_ip):
    reply = ARPMessage.reply(sender_mac, sender_ip, LOCAL_MAC, LOCAL_IP)
    interface.recv_frame(arp_frame(sender_mac, LOCAL_MAC, reply))


def take_arp_request(interface, expected_target_ip):
    """Remove and validate one ARP request, with a useful failure message."""
    frame = interface.maybe_send()
    assert frame is not None, (
        f"Expected an ARP request for unresolved next hop {expected_target_ip}, "
        "but the outgoing queue was empty."
    )
    assert (frame.src, frame.dst, frame.ethertype) == (
        LOCAL_MAC,
        ETHERNET_BROADCAST,
        ETHERTYPE_ARP,
    ), (
        f"Expected a broadcast ARP frame from {LOCAL_MAC}; got "
        f"src={frame.src}, dst={frame.dst}, ethertype={frame.ethertype}."
    )
    message = ARPMessage.parse(frame.payload)
    assert message.opcode == ARP_REQUEST, (
        f"Expected an ARP request for {expected_target_ip}, "
        f"but received opcode {message.opcode}."
    )
    assert message.sender_ip == LOCAL_IP, (
        f"ARP request should identify this interface as sender IP {LOCAL_IP}; "
        f"got {message.sender_ip}."
    )
    assert message.target_ip == expected_target_ip, (
        f"ARP request targeted {message.target_ip}, "
        f"but the unresolved next hop was {expected_target_ip}."
    )
    return message


def take_ipv4_payload(interface, expected_destination_mac, expected_payload):
    """Remove and validate one released IPv4 frame."""
    frame = interface.maybe_send()
    assert frame is not None, (
        f"Expected an IPv4 datagram with payload {expected_payload!r}, "
        "but the outgoing queue was empty."
    )
    assert (frame.src, frame.dst, frame.ethertype) == (
        LOCAL_MAC,
        expected_destination_mac,
        ETHERTYPE_IPV4,
    ), (
        f"Expected IPv4 frame src={LOCAL_MAC}, dst={expected_destination_mac}; "
        f"got src={frame.src}, dst={frame.dst}, ethertype={frame.ethertype}."
    )
    parsed = IPv4Packet.parse(frame.payload)
    assert parsed.payload == expected_payload, (
        f"Expected released payload {expected_payload!r}, "
        f"but got {parsed.payload!r}."
    )
    return parsed


def test_multiple_simultaneous_unresolved_next_hops_keep_state_separate():
    """Each next hop gets one request and only its own datagrams are released."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    datagram_a1 = datagram(b"A1")
    datagram_b1 = datagram(b"B1")
    datagram_a2 = datagram(b"A2")
    datagram_b2 = datagram(b"B2")

    interface.send_datagram(datagram_a1, NEXT_HOP_A)
    interface.send_datagram(datagram_b1, NEXT_HOP_B)
    interface.send_datagram(datagram_a2, NEXT_HOP_A)
    interface.send_datagram(datagram_b2, NEXT_HOP_B)

    take_arp_request(interface, NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_B)
    assert interface.maybe_send() is None, (
        "After one request per unresolved next hop, no duplicate ARP request "
        "should have been queued for either address."
    )

    receive_arp_reply(interface, NEXT_HOP_B_MAC, NEXT_HOP_B)
    take_ipv4_payload(interface, NEXT_HOP_B_MAC, b"B1")
    take_ipv4_payload(interface, NEXT_HOP_B_MAC, b"B2")
    assert interface.maybe_send() is None, (
        "Resolving next hop B must not release datagrams waiting for next hop A."
    )

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"A1")
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"A2")
    assert interface.maybe_send() is None, (
        "All datagrams should be drained after both independent resolutions finish."
    )


def test_several_datagrams_for_one_next_hop_release_in_fifo_order():
    """Waiting datagrams retain send order, including payloads with similar destinations."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    payloads = [b"first", b"second", b"third", b"fourth"]

    for payload in payloads:
        interface.send_datagram(datagram(payload), NEXT_HOP_A)

    take_arp_request(interface, NEXT_HOP_A)
    assert interface.maybe_send() is None, (
        "Additional datagrams for the same unresolved next hop must be appended "
        "to the pending FIFO instead of producing another ARP request."
    )

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    for payload in payloads:
        take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, payload)
    assert interface.maybe_send() is None, (
        "No extra frame should remain after all FIFO datagrams are released."
    )


def test_new_arp_learning_replaces_old_mapping_for_future_datagrams():
    """A later ARP observation must replace the MAC used by subsequent sends."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    interface.send_datagram(datagram(b"using-old-mapping"), NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"using-old-mapping")

    receive_arp_reply(interface, NEXT_HOP_A_MAC_2, NEXT_HOP_A)
    interface.send_datagram(datagram(b"using-replacement"), NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_2, b"using-replacement")
    assert interface.maybe_send() is None, (
        "Replacing a learned mapping should not leave a stale frame queued."
    )


def test_relearning_a_mapping_refreshes_its_cache_lifetime():
    """Learning the same mapping again at 29,999 ms refreshes its 30-second TTL."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)

    interface.tick(29_999)
    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    interface.tick(1)
    interface.send_datagram(datagram(b"still-fresh"), NEXT_HOP_A)

    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"still-fresh")
    assert interface.maybe_send() is None, (
        "A mapping relearned at 29,999 ms must remain usable one millisecond later."
    )


@pytest.mark.parametrize(
    "ethertype,payload",
    [
        (ETHERTYPE_IPV4, b"\x45"),
        (ETHERTYPE_IPV4, b"\x46\x00\x00\x14" + b"\x00" * 16),
        (ETHERTYPE_ARP, b"\x00" * 8),
        (ETHERTYPE_ARP, b"\x00" * 27),
    ],
)
def test_malformed_local_frames_are_ignored_without_side_effects(ethertype, payload):
    """Malformed IPv4 and ARP payloads must not raise, learn, or enqueue frames."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    malformed = EthernetFrame(LOCAL_MAC, OTHER_HOST_MAC, ethertype, payload)

    assert interface.recv_frame(malformed) is None, (
        f"Malformed ethertype {ethertype:#06x} input should be rejected cleanly."
    )
    assert interface.maybe_send() is None, (
        "Rejecting malformed input must not enqueue an ARP reply or any other frame."
    )


def test_frame_unicast_to_another_host_is_ignored_even_when_payload_is_valid():
    """Valid-looking IPv4 and ARP frames for another MAC must not affect us."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    remote_ip = "10.0.0.20"
    remote_mac = "02:00:00:00:00:20"
    valid_ipv4 = EthernetFrame(
        OTHER_HOST_MAC,
        remote_mac,
        ETHERTYPE_IPV4,
        datagram(b"not-for-us").to_bytes(),
    )
    valid_arp = arp_frame(
        remote_mac,
        OTHER_HOST_MAC,
        ARPMessage.reply(remote_mac, remote_ip, OTHER_HOST_MAC, "10.0.0.99"),
    )

    assert interface.recv_frame(valid_ipv4) is None, (
        "A valid IPv4 payload addressed to another host must not reach the IP layer."
    )
    assert interface.recv_frame(valid_arp) is None, (
        "An ARP frame addressed to another host must not be processed or learned."
    )
    interface.send_datagram(datagram(b"requires-resolution"), remote_ip)
    take_arp_request(interface, remote_ip)
    assert interface.maybe_send() is None, (
        "Ignoring another host's ARP frame must leave this next hop unresolved."
    )


def test_malformed_arp_does_not_release_pending_datagrams():
    """A malformed response cannot accidentally flush a pending resolution."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.send_datagram(datagram(b"must-wait"), NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_A)

    malformed_reply = EthernetFrame(
        LOCAL_MAC,
        NEXT_HOP_A_MAC_1,
        ETHERTYPE_ARP,
        b"\x00" * 28,
    )
    assert interface.recv_frame(malformed_reply) is None
    assert interface.maybe_send() is None, (
        "Malformed ARP input must not release the datagram waiting for a valid reply."
    )

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"must-wait")
    assert interface.maybe_send() is None


def test_valid_arp_request_from_another_host_broadcast_to_us_still_learns_and_replies():
    """Broadcast ARP is intentionally accepted, unlike unicast to another host."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    remote_mac, remote_ip = "02:00:00:00:00:20", "10.0.0.20"
    request = ARPMessage.request(remote_mac, remote_ip, LOCAL_IP)

    assert interface.recv_frame(arp_frame(remote_mac, ETHERNET_BROADCAST, request)) is None
    reply_frame = interface.maybe_send()
    assert reply_frame is not None, (
        "A broadcast ARP request for this interface should produce an ARP reply."
    )
    assert reply_frame.dst == remote_mac
    reply = ARPMessage.parse(reply_frame.payload)
    assert (reply.opcode, reply.sender_ip, reply.target_ip) == (
        ARP_REPLY,
        LOCAL_IP,
        remote_ip,
    )
    interface.send_datagram(datagram(b"learned-from-request"), remote_ip)
    take_ipv4_payload(interface, remote_mac, b"learned-from-request")
    assert interface.maybe_send() is None


def test_malformed_input_stays_local_to_that_frame_and_does_not_break_later_processing():
    """After malformed traffic is rejected, a normal resolution still works."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.recv_frame(EthernetFrame(LOCAL_MAC, OTHER_HOST_MAC, ETHERTYPE_ARP, b"bad"))

    interface.send_datagram(datagram(b"normal-after-malformed"), NEXT_HOP_B)
    take_arp_request(interface, NEXT_HOP_B)
    receive_arp_reply(interface, NEXT_HOP_B_MAC, NEXT_HOP_B)
    take_ipv4_payload(interface, NEXT_HOP_B_MAC, b"normal-after-malformed")
    assert interface.maybe_send() is None


def test_unknown_ethertype_to_this_interface_is_ignored_without_side_effects():
    """Only IPv4 and ARP are consumed; an unknown Ethernet payload is inert."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    unknown = EthernetFrame(LOCAL_MAC, OTHER_HOST_MAC, 0x88B5, b"vendor-data")

    assert interface.recv_frame(unknown) is None, (
        "An unsupported ethertype addressed to this interface should be ignored."
    )
    assert interface.maybe_send() is None, (
        "Ignoring an unsupported ethertype must not enqueue a response."
    )


def test_arp_request_for_another_ip_learns_sender_but_does_not_reply():
    """ARP learning is independent from whether the request asks for us."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    remote_mac, remote_ip = "02:00:00:00:00:20", "10.0.0.20"
    request_for_someone_else = ARPMessage.request(
        remote_mac, remote_ip, "10.0.0.99"
    )

    assert interface.recv_frame(
        arp_frame(remote_mac, ETHERNET_BROADCAST, request_for_someone_else)
    ) is None
    assert interface.maybe_send() is None, (
        "A valid ARP request for another IP must not trigger an ARP reply."
    )

    interface.send_datagram(datagram(b"learned-without-reply"), remote_ip)
    take_ipv4_payload(interface, remote_mac, b"learned-without-reply")
    assert interface.maybe_send() is None, (
        "The learned sender mapping should send directly without another request."
    )


def test_unsolicited_arp_reply_learns_mapping_without_creating_output():
    """A valid ARP reply with no pending datagrams updates cache only."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)

    assert interface.maybe_send() is None, (
        "An unsolicited ARP reply should not create an outgoing response."
    )
    interface.send_datagram(datagram(b"uses-unsolicited-learning"), NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"uses-unsolicited-learning")


def test_late_arp_reply_after_pending_timeout_does_not_release_discarded_datagrams():
    """A reply after the five-second wait expires learns only future traffic."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.send_datagram(datagram(b"discard-after-timeout"), NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_A)
    interface.tick(5_000)

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    assert interface.maybe_send() is None, (
        "A reply arriving after pending expiry must not resurrect the discarded "
        "datagram."
    )

    interface.send_datagram(datagram(b"future-traffic"), NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"future-traffic")


def test_pending_timeout_expires_each_next_hop_independently():
    """A timeout removes all old queues but does not merge their replacement queues."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.send_datagram(datagram(b"old-A"), NEXT_HOP_A)
    interface.send_datagram(datagram(b"old-B"), NEXT_HOP_B)
    take_arp_request(interface, NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_B)

    interface.tick(5_000)
    interface.send_datagram(datagram(b"new-A"), NEXT_HOP_A)
    interface.send_datagram(datagram(b"new-B"), NEXT_HOP_B)
    take_arp_request(interface, NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_B)

    receive_arp_reply(interface, NEXT_HOP_B_MAC, NEXT_HOP_B)
    take_ipv4_payload(interface, NEXT_HOP_B_MAC, b"new-B")
    assert interface.maybe_send() is None, (
        "Resolving B after timeout must not release B's expired datagram or A's "
        "new datagram."
    )

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"new-A")
    assert interface.maybe_send() is None


def test_duplicate_arp_replies_do_not_release_pending_datagrams_twice():
    """Once a pending list is flushed, repeated replies cannot duplicate output."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.send_datagram(datagram(b"exactly-once"), NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_A)

    receive_arp_reply(interface, NEXT_HOP_A_MAC_1, NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"exactly-once")
    receive_arp_reply(interface, NEXT_HOP_A_MAC_2, NEXT_HOP_A)

    assert interface.maybe_send() is None, (
        "A second ARP reply may refresh the mapping, but must not resend the "
        "already released datagram."
    )
    interface.send_datagram(datagram(b"after-duplicate"), NEXT_HOP_A)
    take_ipv4_payload(interface, NEXT_HOP_A_MAC_2, b"after-duplicate")


def test_arp_request_can_resolve_pending_datagrams_and_queue_reply_in_order():
    """A request from a waiting next hop releases data before our ARP reply."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    interface.send_datagram(datagram(b"resolved-by-request"), NEXT_HOP_A)
    take_arp_request(interface, NEXT_HOP_A)

    request = ARPMessage.request(NEXT_HOP_A_MAC_1, NEXT_HOP_A, LOCAL_IP)
    interface.recv_frame(arp_frame(NEXT_HOP_A_MAC_1, ETHERNET_BROADCAST, request))

    take_ipv4_payload(interface, NEXT_HOP_A_MAC_1, b"resolved-by-request")
    reply_frame = interface.maybe_send()
    assert reply_frame is not None, (
        "A request for this interface should enqueue an ARP reply after "
        "releasing the pending IPv4 datagram."
    )
    assert (reply_frame.dst, reply_frame.src, reply_frame.ethertype) == (
        NEXT_HOP_A_MAC_1,
        LOCAL_MAC,
        ETHERTYPE_ARP,
    )
    reply = ARPMessage.parse(reply_frame.payload)
    assert (reply.opcode, reply.sender_ip, reply.target_ip) == (
        ARP_REPLY,
        LOCAL_IP,
        NEXT_HOP_A,
    )
    assert interface.maybe_send() is None


def test_broadcast_ipv4_frame_is_parsed_and_returned():
    """Broadcast Ethernet delivery is accepted by the interface's receive path."""
    interface = NetworkInterface(LOCAL_MAC, LOCAL_IP)
    packet = datagram(b"broadcast-payload", destination="255.255.255.255")
    frame = EthernetFrame(
        ETHERNET_BROADCAST,
        OTHER_HOST_MAC,
        ETHERTYPE_IPV4,
        packet.to_bytes(),
    )

    received = interface.recv_frame(frame)
    assert received is not None, "A valid broadcast IPv4 frame should be parsed."
    assert (received.src, received.dst, received.payload) == (
        packet.src,
        packet.dst,
        b"broadcast-payload",
    )
    assert interface.maybe_send() is None, (
        "Receiving broadcast IPv4 data must not create an Ethernet response."
    )

