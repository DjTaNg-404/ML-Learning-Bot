# categories.py - 分类体系定义

# 二级分类体系定义
CATEGORIES = {
    "编程语言基础": {
        "Java核心": ["语法", "集合", "多线程", "JVM", "GC"],
        "Python基础": ["语法", "数据结构", "装饰器", "异步编程"],
        "JavaScript核心": ["语法", "闭包", "原型链", "异步", "ES6+"],
        "C++基础": ["指针", "内存管理", "STL", "面向对象"],
        "Go语言": ["并发", "goroutine", "channel", "包管理"]
    },
    
    "前端技术": {
        "HTML/CSS": ["语义化", "布局", "响应式", "CSS3", "预处理器"],
        "React技术栈": ["组件", "状态", "生命周期", "Hooks", "Redux"],
        "Vue技术栈": ["响应式", "组件通信", "Vuex", "Vue3新特性"],
        "前端工程化": ["Webpack", "构建优化", "模块化", "代码分割"],
        "前端性能": ["加载优化", "渲染优化", "缓存策略"]
    },
    
    "后端框架": {
        "Spring生态": ["IOC", "AOP", "SpringBoot", "SpringCloud"],
        "Node.js": ["Express", "Koa", "中间件", "事件循环"],
        "Django/Flask": ["MVC", "ORM", "中间件", "RESTful API"],
        "微服务架构": ["服务拆分", "注册发现", "负载均衡", "网关"],
        "API设计": ["RESTful", "GraphQL", "接口规范", "版本控制"]
    },
    
    "数据库技术": {
        "MySQL": ["索引", "事务", "锁机制", "查询优化", "主从复制"],
        "Redis": ["数据类型", "持久化", "集群", "缓存策略"],
        "MongoDB": ["文档存储", "聚合查询", "索引", "分片"],
        "数据库设计": ["范式", "ER图", "索引设计", "性能调优"],
        "SQL优化": ["查询计划", "索引使用", "慢查询分析"]
    },
    
    "算法与数据结构": {
        "基础数据结构": ["数组", "链表", "栈", "队列", "哈希表"],
        "树与图": ["二叉树", "平衡树", "图遍历", "最短路径"],
        "排序算法": ["快排", "归并", "堆排序", "时间复杂度分析"],
        "动态规划": ["背包问题", "路径问题", "序列问题"],
        "高频算法题": ["双指针", "滑动窗口", "回溯", "贪心"]
    },
    
    "系统设计": {
        "分布式系统": ["CAP理论", "一致性", "分区容错", "负载均衡"],
        "高并发架构": ["缓存", "消息队列", "读写分离", "分库分表"],
        "设计模式": ["单例", "工厂", "观察者", "策略", "装饰器"],
        "系统架构": ["MVC", "微服务", "事件驱动", "领域驱动"],
        "性能优化": ["并发优化", "内存优化", "网络优化"]
    },
    
    "大数据与AI": {
        "机器学习": ["监督学习", "无监督学习", "特征工程", "模型评估"],
        "深度学习": ["神经网络", "CNN", "RNN", "损失函数", "优化器"],
        "大模型技术": ["Transformer", "Attention", "BERT", "GPT", "微调"],
        "大数据技术": ["Hadoop", "Spark", "Kafka", "数据仓库"],
        "推荐系统": ["协同过滤", "内容推荐", "召回排序", "冷启动"]
    },
    
    "网络与系统": {
        "计算机网络": ["TCP/UDP", "HTTP/HTTPS", "网络模型", "DNS"],
        "操作系统": ["进程线程", "内存管理", "文件系统", "IO模型"],
        "Linux系统": ["命令行", "Shell脚本", "系统监控", "权限管理"],
        "安全技术": ["加密算法", "身份认证", "XSS", "CSRF", "SQL注入"],
        "性能监控": ["APM工具", "日志分析", "指标监控"]
    },
    
    "云服务与运维": {
        "容器技术": ["Docker", "Kubernetes", "容器编排", "镜像管理"],
        "云平台": ["AWS", "阿里云", "服务器配置", "CDN加速"],
        "CI/CD": ["Git工作流", "自动化部署", "测试策略"],
        "监控运维": ["日志收集", "性能监控", "告警机制", "故障排查"],
        "DevOps": ["基础设施即代码", "自动化运维", "发布策略"]
    },
    
    "项目经验": {
        "项目架构": ["技术选型", "架构设计", "模块划分"],
        "问题解决": ["性能瓶颈", "bug排查", "优化方案"],
        "团队协作": ["代码规范", "版本管理", "code review"],
        "业务理解": ["需求分析", "用户体验", "产品思维"],
        "技术管理": ["技术决策", "风险评估", "技术债务"]
    }
}

# 辅助功能函数
def get_all_primary_categories():
    """获取所有一级分类"""
    return list(CATEGORIES.keys())

def get_secondary_categories(primary_category):
    """获取指定一级分类下的所有二级分类"""
    return list(CATEGORIES.get(primary_category, {}).keys())

def get_keywords(primary_category, secondary_category):
    """获取指定分类的关键词列表"""
    return CATEGORIES.get(primary_category, {}).get(secondary_category, [])

def find_category_by_keywords(keywords):
    """根据关键词查找可能的分类"""
    matches = []
    for primary, secondary_dict in CATEGORIES.items():
        for secondary, keyword_list in secondary_dict.items():
            for keyword in keywords:
                if keyword.lower() in [k.lower() for k in keyword_list]:
                    matches.append((primary, secondary))
                    break
    return matches

def validate_category(primary_category, secondary_category):
    """验证分类是否有效"""
    return (primary_category in CATEGORIES and 
            secondary_category in CATEGORIES.get(primary_category, {}))

# 用于LLM提示词的分类说明
CATEGORY_DESCRIPTIONS = {
    "编程语言基础": "编程语言的核心语法、特性和基础概念",
    "前端技术": "Web前端开发相关的技术栈和工具",
    "后端框架": "服务端开发框架和架构设计",
    "数据库技术": "数据存储、查询优化和数据库设计",
    "算法与数据结构": "计算机科学基础算法和数据结构",
    "系统设计": "大型系统架构设计和分布式系统",
    "大数据与AI": "机器学习、深度学习和大数据处理",
    "网络与系统": "计算机网络、操作系统和安全技术", 
    "云服务与运维": "云计算、容器化和DevOps实践",
    "项目经验": "实际项目开发和团队协作经验"
}

# 难度等级定义
DIFFICULTY_LEVELS = {
    "初级": "基础概念和语法，适合应届生",
    "中级": "实际应用和原理理解，适合1-3年经验",
    "高级": "深入原理和架构设计，适合3-5年经验", 
    "专家级": "系统设计和创新方案，适合5年以上经验"
}

# 问题类型定义
QUESTION_TYPES = {
    "概念题": "理论知识和概念理解",
    "代码题": "现场编程和算法实现",
    "设计题": "系统架构和方案设计",
    "经验题": "项目经历和实践分享",
    "场景题": "问题分析和解决方案"
}

def generate_category_description():
    """为LLM生成分类体系描述"""
    description = "分类体系如下：\n\n"
    
    # 一级分类说明
    description += "【一级分类】\n"
    for primary, desc in CATEGORY_DESCRIPTIONS.items():
        description += f"- {primary}：{desc}\n"
    
    description += "\n【二级分类示例】\n"
    # 只显示几个主要的二级分类示例，避免提示词过长
    for primary, secondary_dict in list(CATEGORIES.items())[:5]:  # 只显示前5个
        secondary_list = list(secondary_dict.keys())[:3]  # 每个只显示前3个
        description += f"- {primary}：{', '.join(secondary_list)}等\n"
    
    description += "\n【难度等级】\n"
    for level, desc in DIFFICULTY_LEVELS.items():
        description += f"- {level}：{desc}\n"
    
    return description