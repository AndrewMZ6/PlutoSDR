import threading
from time import sleep



def task(lock):
    thread_name = threading.current_thread()

    for i in range(5):
        print(f"thread {thread_name} without LOCK")
        sleep(1)

    with lock:
        for i in range(10):
            print(f"thread {thread_name} uses 'with lock'")
            sleep(1)
    print(f"thread {thread_name} is finished")


def counter(lock):
    thread_name = threading.current_thread()
    for _ in range(10):
        print(f"thread -> {thread_name}, iter -> {_}")
        sleep(1)



lock = threading.Lock()

t1 = threading.Thread(target=task, args=(lock, ))
t1.start()

t2 = threading.Thread(target=counter, args=(lock, ))
t2.start()

'''
thread_name = threading.current_thread()
for i in range(5):
    print(f"thread {thread_name} uses 'with lock'")
    sleep(1)
'''

t1.join()
t2.join()