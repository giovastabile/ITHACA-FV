import time
import functools

def clock(func):
    @functools.wraps(func)
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter()-t0
        name=func.__qualname__
        arg_str=','.join(repr(arg) for arg in args)
        print('[%0.8fs] %s' % (elapsed, name))
        # print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked

@clock
def snooze(seconds):
    time.sleep(seconds)

@clock
def factorial(n):
    return 1 if n<3 else n*factorial(n-1)

@functools.lru_cache()
@clock
def fibonacci(n):
    if n<2:
        return n
    return fibonacci(n-2)+fibonacci(n-1)


if __name__=='__main__':
    print('*'*40, 'calling snooze(.123)')
    snooze(.123)
    print('*'*40, 'Calling factorial(6)')
    print('6!=', factorial(6))
    print(fibonacci(6), fibonacci.__name__)