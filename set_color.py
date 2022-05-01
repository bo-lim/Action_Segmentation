set_color = {
    'action_start':(204,204,204),
    # cut_and_mix_ingredients
    'peel_cucumber':(51,102,51),
    'cut_cucumber':(204,255,204),
    'place_cucumber_into_bowl':(255,102,255),
    'cut_tomato':(255,51,51),
    'place_tomato_into_bowl':(255,102,153),
    'cut_cheese':(204,153,10),
    'place_cheese_into_bowl':(255,205,0),
    'cut_lettuce':(0,153,0),
    'place_lettuce_into_bowl':(51,255,51),
    'mix_ingredients':(102,0,255),
    # prepare_dressing
    'add_oil':(153,102,102),
    'add_vinegar':(255,204,153),
    'add_salt':(153,153,153),
    'add_pepper':(51,51,0),
    'mix_dressing':(51,255,255),
    # serve_salad
    'serve_salad_onto_plate':(0,51,153),
    'add_dressing':(51,153,204),
    'action_end':(204,255,255),
}
def get_color(action):
    return set_color[action]