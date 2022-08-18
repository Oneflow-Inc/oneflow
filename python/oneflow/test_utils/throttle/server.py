from redis import Redis
import time

def main():
    redis = Redis()
    print(redis.connection_pool.connection_kwargs['port'])
    while(1):
        time.sleep(3)


if __name__ == "__main__":
    main()
