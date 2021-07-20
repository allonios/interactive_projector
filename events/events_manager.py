from event_bus import EventBus

bus = EventBus()


@bus.on("clicked")
def consume_click(event_data):
    print("consumed click for:", event_data)

