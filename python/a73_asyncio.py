import asyncio
import time


async def hello():  # async def로 네이티브 코루틴을 만듦
    print("Hello, world!")
    time.sleep(5)


def hello2():  # async def로 네이티브 코루틴을 만듦
    print("Hello, world!")
    time.sleep(5)


loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
loop.run_until_complete(hello())  # hello가 끝날 때까지 기다림
hello2()  # hello가 끝난 후에 실행됨
print("end")
loop.close()  # 이벤트 루프를 닫음
