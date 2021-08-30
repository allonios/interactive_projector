from event_bus import EventBus

bus = EventBus()


@bus.on("clicked")
def consume_click(event_data):
    print("consumed click for:")
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(event_data)
    hand_x = event_data["hand_center"][0]
    hand_y = event_data["hand_center"][1]
    print(f"hand x: {hand_x}, hand y: {hand_y}")

    # dragTo(hand_x, hand_y, button=LEFT)
