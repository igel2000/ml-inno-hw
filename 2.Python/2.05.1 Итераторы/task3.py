from string import ascii_lowercase, ascii_uppercase
import random
def password_generator(password_length=12, seed=None):
    chars = ascii_lowercase + ascii_uppercase + "0123456789!?@#$*"
    if seed is not None and isinstance(seed, int):
        random.seed(seed)
    while True:
        password = [chars[random.randint(0, len(chars) - 1)] for _ in range(password_length)]
        yield "".join(password)

