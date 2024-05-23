from polosdk import RestClient
import time
import data_utils

# 获取波场连接客户端
client = RestClient()

# 0 low	                String	lowest price over the interval
# 1 high	            String	highest price over the interval
# 2 open	            String	price at the start time
# 3 close	            String	price at the end time
# 4 amount	            String	quote units traded over the interval
# 5 quantity	        String	base units traded over the interval
# 6 buyTakerAmount	    String	quote units traded over the interval filled by market buy orders
# 7 buyTakerQuantity	String	base units traded over the interval filled by market buy orders
# 8 tradeCount	        Integer	count of trades
# 9 ts	                Long	time the record was pushed
# 10 weightedAverage	String	weighted average over the interval
# 11 interval	        String	the selected interval
# 12 startTime	        Long	start time of interval
# 13 closeTime	        Long	close time of interval

# 获取当前时间戳
timestamp = int(time.time())
print(timestamp)

for j in range(0,980,10):
    '''
        从symbol表中获取所有的虚拟币
        调用波场接口获取该虚拟币的交易记录，并存入history表
    '''
    limit = 10
    j = j + limit
    # 获取虚拟币列表
    symbols = data_utils.get_symbol(start=j, limit=limit)
    for symbol in symbols: 
        try:
            coin_name = symbol[1]
            print(coin_name, j)
            # 波场接口每次获取100条数据，循环获取交易记录
            for i in range(15):
                step = 60 * 100 * i
                end = timestamp - step
                start = end - 6000
                # print(start, end)
                charts = client.markets().get_candles(coin_name, 'MINUTE_1', start * 1000, end * 1000)
                # print(charts)
                data_utils.fill_part_data(coin_name, charts)
        except Exception as e:
            # print(e)
            continue
            # break

print(timestamp)
