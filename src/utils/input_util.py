def get_choice(text, choices):
    choice = input(text)
    while choice not in choices:
        print(f'Invalid choice: {choice}. Select one of {choices}.')
        choice = input(text)
    return choice
