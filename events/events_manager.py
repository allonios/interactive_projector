from event_bus import EventBus
from pyautogui import LEFT, dragTo

bus = EventBus()


@bus.on("clicked")
def consume_click(event_data):
    print("consumed click for:")
    print(event_data)

    click_coords_x = event_data["hand_coords"][0]
    click_coords_y = event_data["hand_coords"][1]

    dragTo(int(click_coords_x), int(click_coords_y), button=LEFT)
