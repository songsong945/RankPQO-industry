import pymysql

# 连接到 OceanBase 数据库服务器
conn = pymysql.connect(
    host="localhost",
    port=10400,  # 请根据您的实际端口号进行修改
    user="root",
    passwd=""
)

try:
    with conn.cursor() as cur:
        # 创建数据库 testdb，如果不存在
        cur.execute("CREATE DATABASE IF NOT EXISTS testdb")
        # 选择数据库 testdb
        cur.execute("USE testdb")
        # 创建表 cities，如果不存在
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cities (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                population INT
            )
        """)
        # 插入示例数据
        cur.execute("INSERT INTO cities (name, population) VALUES ('CityA', 1000000)")
        cur.execute("INSERT INTO cities (name, population) VALUES ('CityB', 500000)")
        conn.commit()

        # 查询数据
        cur.execute("SELECT * FROM cities")
        results = cur.fetchall()
        for row in results:
            print(row)
finally:
    conn.close()
