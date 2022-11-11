import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

BG_COLOR = (7, 99, 36)
WHITE = (255, 255, 255)
RED = (217, 30, 61)
CHIPS = {
    0: {"value": 10000, "img": "ChipOrange", "number": 0},
    1: {"value": 5000, "img": "ChipPink", "number": 0},
    2: {"value": 1000, "img": "ChipYellow", "number": 0},
    3: {"value": 100, "img": "ChipBlack", "number": 0},
    4: {"value": 50, "img": "ChipBlue", "number": 0},
    5: {"value": 25, "img": "ChipGreen", "number": 0},
    6: {"value": 10, "img": "ChipLightBlue", "number": 0},
    7: {"value": 5, "img": "ChipRed", "number": 0},
    8: {"value": 1, "img": "ChipWhite", "number": 0},
}

HERE = Path(__file__).parent.resolve()
SPRITE_PATH = HERE / "sprites"


def get_image(image_name: str) -> pygame.surface.Surface:
    return pygame.image.load(SPRITE_PATH / (image_name + ".png"))


def get_tile_size(screen_height) -> float:
    return screen_height / 5


def render(
    screen: pygame.Surface,
    player_states: Dict[str, Dict],
    most_recent_move: Dict,
    render_opponent_cards: bool = True,
    show_player_names: bool = True,
    continue_hands: bool = True,
    win_message: Optional[str] = None,
) -> None:
    """player_states is a dict of player_id: player_state."""

    # Use me for external screen injection
    # mode = "human"
    screen_height = screen.get_height()
    screen_width = screen.get_width()
    tile_size = get_tile_size(screen_height)

    # Setup dimensions for card size and setup for colors
    screen.fill(BG_COLOR)

    # Load and blit all images for each card in each player's hand

    for i, (name, state) in enumerate(player_states.items()):
        for j, card in enumerate(state["hand"]):
            if not render_opponent_cards and name == "opponent":
                card_img = get_image("Card")
            else:
                card_img = get_image(card)
            card_img = pygame.transform.scale(
                card_img, (int(tile_size * (142 / 197)), int(tile_size))
            )
            # Players with even id go above public cards
            if i % 2 == 0:
                screen.blit(
                    card_img,
                    (
                        (
                            calculate_card_width(
                                idx=i, screen_width=screen_width, tile_size=tile_size
                            )
                            - calculate_offset(state["hand"], j, tile_size)
                        ),
                        calculate_height(screen_height, 4, 1, tile_size, -1),
                    ),
                )
            # Players with odd id go below public cards
            else:
                screen.blit(
                    card_img,
                    (
                        (
                            calculate_card_width(
                                idx=i, screen_width=screen_width, tile_size=tile_size
                            )
                            - calculate_offset(state["hand"], j, tile_size)
                        ),
                        calculate_height(screen_height, 4, 3, tile_size, 0),
                    ),
                )

        # Load and blit text for player name
        font = pygame.font.SysFont("arial", 22)
        move_map: Dict[int, str] = {
            None: "",
            0: "fold",
            1: "call/check",
            2: "raise",
            3: "raise",
            4: "all in",
        }  # type: ignore

        move = move_map[most_recent_move[i]]

        if show_player_names:
            text = font.render(f"{name}: move = {move}", True, WHITE)
        else:
            text = font.render(f"move = {move}", True, WHITE)

        textRect = text.get_rect()
        if i % 2 == 0:
            textRect.center = (
                (screen_width / (np.ceil(len(player_states) / 2) + 1) * np.ceil((i + 1) / 2)),
                calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)),
            )
        else:
            textRect.center = (
                (screen_width / (np.ceil(len(player_states) / 2) + 1) * np.ceil((i + 1) / 2)),
                calculate_height(screen_height, 4, 3, tile_size, (23 / 20)),
            )
        screen.blit(text, textRect)

        x_pos, y_pos = get_player_chip_position(
            player_idx=i,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        draw_chips(screen=screen, x_pos=x_pos, y_pos=y_pos, n_chips=state["my_chips"])

    # Load and blit public cards
    public_cards = player_states["player"]["public_cards"]
    for i, card in enumerate(public_cards):
        card_img = get_image(card)
        card_img = pygame.transform.scale(card_img, (int(tile_size * (142 / 197)), int(tile_size)))
        if len(public_cards) <= 3:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(public_cards, i, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)),
                    )
                ),
            )
        elif i <= 2:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(state["public_cards"][:3], i, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, -21 / 20),
                    )
                ),
            )
        else:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(public_cards[3:], i - 3, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, 1 / 20),
                    )
                ),
            )

    if win_message is not None:
        # Load and blit text for player name
        font = pygame.font.SysFont("arial", 22)
        text = font.render(win_message, True, RED)
        textRect = text.get_rect()
        textRect.center = (screen_width // 2, int(screen_height * 0.45))
        pygame.draw.rect(screen, WHITE, textRect)
        screen.blit(text, textRect)

        second_line = "Click to continue" if continue_hands else None

        text = font.render(second_line, True, RED)
        textRect = text.get_rect()
        textRect.center = (
            screen_width // 2,
            int(screen_height * 0.55),
        )
        pygame.draw.rect(screen, WHITE, textRect)
        screen.blit(text, textRect)

    # pygame.display.update()


def calculate_card_width(idx: int, screen_width: int, tile_size: float, n_agents: int = 2) -> int:
    return int(
        (screen_width / (np.ceil(n_agents / 2) + 1) * np.ceil((idx + 1) / 2)) + tile_size * 31 / 616
    )


def get_player_chip_position(
    player_idx: int, screen_width: int, screen_height: int
) -> Tuple[int, int]:

    tile_size = get_tile_size(screen_height)

    if player_idx % 2 == 0:
        offset = -0.2
        multiplier = 1
    else:
        offset = 0.5
        multiplier = 3

    x_pos = calculate_card_width(
        player_idx, screen_width=screen_width, tile_size=tile_size
    ) + tile_size * (8 / 10)
    y_pos = calculate_height(screen_height, 4, multiplier, tile_size, offset)
    return (x_pos, y_pos)


def calculate_height(
    screen_height: int, divisor: float, multiplier: float, tile_size: float, offset: float
) -> int:
    return int(multiplier * screen_height / divisor + tile_size * offset)


def calculate_offset(hand: List, j: int, tile_size: float):
    return int((len(hand) * (tile_size * 23 / 56)) - ((j) * (tile_size * 23 / 28)))


def draw_chips(
    screen: pygame.surface.Surface,
    n_chips: int,
    x_pos: int,
    y_pos: int,
):

    font = pygame.font.SysFont("arial", 20)
    text = font.render(str(n_chips), True, WHITE)
    textRect = text.get_rect()
    tile_size = get_tile_size(screen.get_height())

    # Calculate number of each chip
    height = 0

    # Draw the chips
    for key in CHIPS:
        num = n_chips / CHIPS[key]["value"]
        CHIPS[key]["number"] = int(num)
        n_chips %= CHIPS[key]["value"]

        chip_img = get_image(CHIPS[key]["img"])
        chip_img = pygame.transform.scale(chip_img, (int(tile_size / 2), int(tile_size * 16 / 45)))

        for j in range(int(CHIPS[key]["number"])):
            height_offset = (j + height) * tile_size / 15
            screen.blit(
                chip_img,
                (x_pos, y_pos - height_offset),
            )
        height += CHIPS[key]["number"]

    # Blit number of chips
    textRect.center = (
        x_pos + tile_size // 4,
        y_pos - ((height + 1) * tile_size / 15),
    )
    screen.blit(text, textRect)
