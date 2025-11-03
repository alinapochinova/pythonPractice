def my_enumerate(iterable, start=0):
    """
    Аналог встроенной функции enumerate.

    Args:
        iterable: итерируемый объект
        start: начальное значение счетчика (по умолчанию 0)

    Yields:
        кортеж (index, element) для каждого элемента iterable
    """
    index = start
    for element in iterable:
        yield index, element
        index += 1


def fibonacci():
    """
    Генератор бесконечной последовательности чисел Фибоначчи.

    Yields:
        следующее число Фибоначчи в последовательности
    """
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def my_cycle(iterable):
    """
    Бесконечно повторяет элементы iterable.

    Args:
        iterable: итерируемый объект

    Yields:
        элементы iterable в бесконечном цикле
    """
    while True:
        for element in iterable:
            yield element


def my_repeat(obj, times=None):
    """
    Повторяет объект указанное количество раз или бесконечно.

    Args:
        obj: объект для повторения
        times: количество повторений (None для бесконечного повторения)

    Yields:
        объект obj указанное количество раз
    """
    if times is None:
        while True:
            yield obj
    else:
        for _ in range(times):
            yield obj


def my_product(*iterables):
    """
    Декартово произведение входных итерируемых объектов.

    Args:
        *iterables: произвольное количество итерируемых объектов

    Yields:
        кортежи, представляющие все возможные комбинации элементов
    """
    if not iterables:
        yield ()
        return

    def product_helper(current, remaining):
        if not remaining:
            yield current
        else:
            for item in remaining[0]:
                yield from product_helper(current + (item,), remaining[1:])

    yield from product_helper((), iterables)


def my_product_iterative(*iterables):
    """
    Итеративная реализация декартова произведения.
    """
    pools = [list(pool) for pool in iterables]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


print("Пример работы my_enumerate:")
fruits = ['apple', 'banana', 'cherry']
for i, fruit in my_enumerate(fruits, 1):
    print(f"{i}: {fruit}")

print("\nФибоначчи(первые 10 чисел):")
fib_gen = fibonacci()
for i in range(10):
    print(next(fib_gen), end=" ")
print()

print("\nПример работы my_cycle(первые 8 элементов):")
cycle_gen = my_cycle(['A', 'B', 'C'])
for i in range(8):
    print(next(cycle_gen), end=" ")
print()

print("\nПример работы my_repeat:")
repeat_gen = my_repeat('hello', 3)
print(list(repeat_gen))

print("\nПример работы my_product для получения product([1,2], ['a','b']):")
for item in my_product([1, 2], ['a', 'b']):
    print(item)

print("\nПример использования my_product для получения product(range(2), repeat=2):")
for item in my_product(range(2), range(2)):
    print(item)