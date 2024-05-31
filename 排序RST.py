import os
import struct
import time
import numpy as np
from Curves import *

# QUERY_FILE = r'E:\Python_Projects\RTree\查询\6万查询.txt'
QUERY_FILE = r'E:\Python_Projects\RTree\查询\point_query.txt'
# QUERY_FILE = r'E:\Python_Projects\RTree\查询\query_uniform.txt'

# INDEX_FILE = r'E:\Python_Projects\RTree\index\index6.idx'
INDEX_FILE = r'E:\Python_Projects\RTree\index\index_800W_uni.idx'
ORI_FILE = r'E:\Python_Projects\RTree\data\data.txt'
D_FILE = ORI_FILE.split('.')[0] + '.dat'


DB_COUNT = 0
point_count = 0
BLOCK_SIZE = 4 * 1024  # 块大小
IO_COUNT = 0  # 计算IO次数
BUFFER_SIZE = 100000000000000  # 缓冲容量
MAX_ITEM_AMOUNT = (BLOCK_SIZE-45)//36  # 索引块最大容量


def str_sort(file=ORI_FILE):
    """利用str算法排序数据"""
    datapoints = []
    with open(ORI_FILE, 'r') as f:
        for line in f:
            x, y = line.split(',')
            datapoints.append([float(x), float(y)])
    np_data = np.array(datapoints)
    # 先按照x轴排序
    data_sorted_x = np_data[np_data[:, 0].argsort()]
    # 数据块最大容量
    dblock_cap = (BLOCK_SIZE-8)//16
    # 叶子数目
    P = np.ceil(np_data.shape[0] // dblock_cap)
    # 切片数目
    S = np.ceil(np.sqrt(P))
    # 除了最后一个切片，每个切片包含的数据数
    data_num_in_slice = S * dblock_cap
    split_data = [data_sorted_x[i:i+int(data_num_in_slice)] for i in range(0, len(data_sorted_x), int(data_num_in_slice))]

    # 在每个分组内按照y轴排序，并合并数据
    final_sorted_data = []
    for group in split_data:
        group_sorted_y = group[group[:, 1].argsort()]
        final_sorted_data.append(group_sorted_y)
    final_sorted_data = np.concatenate(final_sorted_data)
    data = list(final_sorted_data)
    return data


class RTree:
    def __init__(self, block_id, mbr=None):
        self.block_id = block_id
        if mbr == None:
            self.mbr = RTree.init_MBR()
        else:
            self.mbr = mbr
        self.item_list = []

    def is_intersect(self, mbr):
        """
        判断相交
        :param mbr:
        :return:
        """
        return self.mbr[0] <= mbr[2] and self.mbr[2] >= mbr[0]\
            and self.mbr[1] <= mbr[3] and self.mbr[3] >= mbr[1]

    def update_Mbr(self):
        """
        插入数据更新当前节点mbr
        :param mbr:
        :return:
        """
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        for item in self.item_list:
            x1_list.append(item[0][0])
            x2_list.append(item[0][2])
            y1_list.append(item[0][1])
            y2_list.append(item[0][3])
        self.mbr[0] = min(x1_list)
        self.mbr[2] = max(x2_list)
        self.mbr[1] = min(y1_list)
        self.mbr[3] = max(y2_list)

    @staticmethod
    def init_MBR():
        return [float('inf'), float('inf'), float('-inf'), float('-inf')]


class Header(RTree):
    """头结点，保存元数据，块号默认是0"""
    def __init__(self, begin_offset, block_sum, ptr=0, mbr=None, block_id=0, block_size=BLOCK_SIZE):
        RTree.__init__(self,block_id,mbr)
        self.block_sum = block_sum
        self.begin_offset = begin_offset
        self.ptr = ptr
        self.block_size = block_size

    def update_mbr(self, mbr):
        x1 = min(self.mbr[0], mbr[0])
        y1 = min(self.mbr[1], mbr[1])
        x2 = max(self.mbr[2], mbr[2])
        y2 = max(self.mbr[3], mbr[3])
        return [x1, y1, x2, y2]

    def pack(self) -> bytes:
        """
        头信息->二进制信息
        :return:
        """
        return struct.pack('4d5I', *self.mbr, self.begin_offset, self.ptr
                           , self.block_sum, self.block_id, self.block_size)

    @staticmethod
    def unpack(bin_data):
        """
        二进制信息->头信息
        :return:
        """
        x1, y1, x2, y2, begin_offset, ptr, block_sum, block_id, block_size = struct\
        .unpack('4d5I', bin_data[:52])
        return Header(mbr=[x1, y1, x2, y2], begin_offset=begin_offset,
                      ptr=ptr, block_sum=block_sum, block_size=block_size)

    @staticmethod
    def load_Header(index_file):
        if not os.path.exists(index_file):
            with open(index_file, 'wb'):
                pass
                # 文件不存在，创建文件
        with open(index_file,'rb') as fp:
            fp.seek(0)
            bin_data = fp.read(BLOCK_SIZE)
            if len(bin_data) == 0:
                return None
            else:
                return Header.unpack(bin_data)

    @staticmethod
    def update_Header(header,index_file):
        with open(index_file, 'rb+') as fp:  # wb会覆盖原来的数据
            fp.seek(0)
            fp.write(header.pack())


class LeafNode(RTree):
    def __init__(self, block_id, mbr=None, is_leaf=True, parent_id=0):
        RTree.__init__(self,block_id,mbr)
        self.item_list = []  # 保存形式为mbr,dblock_id
        self.is_leaf = is_leaf
        self.dblock_id_list = []  # 存储数据块ID
        self.parent_id = parent_id

    def add_item(self, entry):  # entry:[mbr, dblock_id]
        """
        叶子结点添加项目，如果数据块号不在列表中则加入列表，否则更新数据块号对应
        数据块的mbr信息
        :param entry:
        :return:
        """
        if entry[1] not in self.dblock_id_list:
            self.item_list.append([entry[0],entry[1]])
            self.dblock_id_list.append(entry[1])
        else:
            for i in range(len(self.item_list)):
                if self.item_list[i][1] == entry[1]:
                    self.item_list[i][0] = entry[0]
        # self.item_list.append([mbr, entry[1]])
        self.update_Mbr()
    # def delete_item(self,):

    def is_overflow(self):
        return (len(self.item_list)*36) + 45 > BLOCK_SIZE

    def is_underflow(self):
        return len(self.item_list) < (MAX_ITEM_AMOUNT)//2

    def pack(self):
        """
        块信息->二进制信息
        :return:
        """
        # 注：打包时d要在I前面，不然会出现内存对齐问题（或者以>方式打包）
        block_info = struct.pack('4d3I?', *self.mbr, self.block_id,
                                 self.parent_id, len(self.item_list), self.is_leaf)
        for mbr,dblock_id in self.item_list:
            block_info += struct.pack('4dI', *mbr, dblock_id)
        return block_info

    @staticmethod
    def unpack(bin_data):
        """
        二进制信息->块信息
        :param bin_data:
        :return:
        """
        x1, y1, x2, y2, block_id, \
            parent_id, len, is_leaf \
            = struct.unpack('4d3I?', bin_data[:45])
        leaf_node = LeafNode(block_id=block_id, mbr=[x1, y1, x2, y2], parent_id=parent_id \
                               , is_leaf=is_leaf)
        offset = 45
        for i in range(len):
            mbr = list(struct.unpack('4d', bin_data[offset:offset + 32]))
            offset += 32
            dblock_id = (struct.unpack('I', bin_data[offset:offset + 4]))[0]
            offset += 4
            leaf_node.item_list.append([mbr, dblock_id])
        return leaf_node


class InterNode(RTree):
    is_leaf = False

    def __init__(self, block_id, mbr=None, is_leaf = False, parent_id = 0):
        RTree.__init__(self, block_id, mbr)
        self.is_leaf = is_leaf
        self.parent_id = parent_id
        self.item_list = []  # mbr,block_id

    def add_item(self, entry):
        self.item_list.append([entry[0],entry[1]])
        self.update_Mbr()

    def is_overflow(self):
        """
        判断上溢出
        :return:
        """
        return len(self.item_list)*36 + 45 > BLOCK_SIZE

    def is_underflow(self):
        """
        判断下溢
        :return:
        """
        return len(self.item_list) < (MAX_ITEM_AMOUNT)//2

    def pack(self):
        block_info = struct.pack('4d3I?', *self.mbr, self.block_id,
                                 self.parent_id, len(self.item_list), self.is_leaf)
        for mbr, block_id in self.item_list:
            block_info += struct.pack('4dI', *mbr, block_id)
        return block_info

    @staticmethod
    def unpack(bin_data):
        """
        注：struct中的unpack返回的是一个元祖
        :param bin_data:
        :return:
        """
        x1,y1,x2,y2, block_id, \
            parent_id, len, is_leaf\
        = struct.unpack('4d3I?',bin_data[:45])
        inter_node = InterNode(block_id=block_id, mbr=[x1,y1,x2,y2], parent_id=parent_id
                         ,is_leaf=False)
        offset = 45
        for i in range(len):
            mbr = list(struct.unpack('4d', bin_data[offset:offset+32]))
            offset += 32
            block_id = (struct.unpack('I', bin_data[offset:offset+4]))[0]
            offset += 4
            inter_node.item_list.append([mbr, block_id])
        return inter_node


class Operation:
    def __init__(self, indexcache, datacache):
        self.indexcache = indexcache
        self.datacache = datacache

    @staticmethod
    def load_block(index_file, block_id):
        """
        从文件加载索引块
        :param index_file: 索引文件
        :param block_id: 块号
        :return:
        """
        with open(index_file,'rb') as fp:
                fp.seek(block_id*BLOCK_SIZE)
                bin_data = fp.read(BLOCK_SIZE)
                type = struct.unpack('?',bin_data[44:45])[0]
                if type:
                    return LeafNode.unpack(bin_data)
                else:
                    return InterNode.unpack(bin_data)


class Query(Operation):
    def __init__(self,index,indexcache, datacache, query_file=QUERY_FILE,indexfile=INDEX_FILE):
        Operation.__init__(self,indexcache,datacache)
        self.index = index
        self.query_file = query_file
        self.indexfile = indexfile

    def point_query(self, point):
        """
        点查询
        :param point:查询点
        :return:
        """
        result = set() # 临时结果集
        # result = [] # 临时结果集
        global IO_COUNT
        header = self.indexcache.get_Block(0)
        if header == None:
            header = Header.load_Header(self.indexfile)
            self.indexcache.add_Block(header)
        root_id = header.ptr
        root = self.indexcache.get_Block(root_id)
        if root == None:
            root = self.load_block(INDEX_FILE, root_id)
            IO_COUNT += 1
            self.indexcache.add_Block(root)

        # 最小mbr包含查询点的索引节点列表
        contained_node_list = []

        if self.is_contained(root.mbr, point):
            contained_node_list.append(root)

        while len(contained_node_list) > 0:  # 层次遍历
            node = contained_node_list.pop(0)
            if node.is_leaf:  # 处理叶子
                for mbr, dblock_id in node.item_list:
                    if self.is_contained(mbr, point):
                        dblock = self.datacache.get_dblock(dblock_id)
                        if dblock == None:
                            dblock = Data.load_dblock(dblock_id)
                            IO_COUNT += 1
                            self.datacache.add_dblock(dblock)
                        for x, y in dblock.data_points:
                            if x == point[0] and y == point[1]:
                                result.add((x, y))
                                # result.append((poi.x,poi.y))
                                return result
            else:  # 处理内部节点
                for mbr, block_id in node.item_list:
                    if self.is_contained(mbr, point):
                        tmpnode = self.indexcache.get_Block(block_id)
                        if tmpnode == None:
                            tmpnode = self.load_block(INDEX_FILE,block_id)
                            IO_COUNT += 1
                            self.indexcache.add_Block(tmpnode)
                        contained_node_list.append(tmpnode)

    def range_query(self, range):
        """
        范围查询
        :param range:查询范围
        :return: 返回在范围内的点
        """

        result_dblock_id_set = set()  # 和查询范围有交集的数据块号集合
        result = set() # 结果集
        # result = [] # 结果集
        overlap_node_list = []  # 和查询有交集的索引结点id

        global IO_COUNT
        header = self.indexcache.get_Block(0)
        if header == None:
            header = Header.load_Header(self.indexfile)
            self.indexcache.add_Block(header)
        root_id = header.ptr
        root = self.indexcache.get_Block(root_id)
        if root == None:
            IO_COUNT += 1
            root = self.load_block(self.indexfile, root_id)
            self.indexcache.add_Block(root)
        if root.is_intersect(range) == False:
            print(f'error')
            return []
        else:
            overlap_node_list.append(root)

        while len(overlap_node_list) > 0:  # 层次遍历
                node = overlap_node_list.pop(0)
                if node.is_leaf:  # 检测叶子结点
                    for mbr,dblock_id in node.item_list:
                        if self.is_overlap(range,mbr):
                            result_dblock_id_set.add(dblock_id)
                else:  # 检测内部节点
                    for mbr, block_id in node.item_list:
                        if self.is_overlap(range,mbr):
                            tmpnode = self.indexcache.get_Block(block_id)
                            if tmpnode == None:
                                tmpnode = self.load_block(INDEX_FILE,block_id)
                                IO_COUNT += 1
                                self.indexcache.add_Block(tmpnode)
                            overlap_node_list.append(tmpnode)

        # 获取结果数据块，然后扫描其中的数据点是否在查询范围内
        for dblock_id in result_dblock_id_set:
            dblock = self.datacache.get_dblock(dblock_id)
            if dblock == None:
                dblock = Data.load_dblock(dblock_id)
                IO_COUNT += 1
                self.datacache.add_dblock(dblock)
            for x, y in dblock.data_points:
                if self.is_contained(range, [x, y]):
                    result.add((x,y))
                    # result.append((point.x, point.y))
        return result

    def is_overlap(self, query, mbr):
        """
        判断相交
        :param mbr1:
        :param mbr2:
        :return:
        """
        return query[0] <= mbr[2] and query[1] <= mbr[3] \
                and query[2] >= mbr[0] and query[3] >= mbr[1]

    def is_contained(self, query, point):
        """
        判断查询点是否在矩形内
        :param query:
        :param point:
        :return:
        """
        return query[0] <= point[0] <= query[2] and\
        query[1] <= point[1] <= query[3]


class Index(Operation):
    def __init__(self,indexcache, datacache, index_file= INDEX_FILE):
        Operation.__init__(self,indexcache, datacache)
        self.index_file = index_file

    def create_Index(self,entry):
        """
        开始建立索引
        :param entry:
        :return:
        """
        global IO_COUNT
        header = self.indexcache.get_Block(block_id=0)
        if header == None:
            header = Header.load_Header(self.index_file)
            if header == None:
                header = Header(mbr = Header.init_MBR(), begin_offset=4096,
                                block_size=BLOCK_SIZE, block_sum=1)  # 头结点算一块
                # Header.update_Header(header, self.index_file)
                # IO_COUNT += 1
        self.indexcache.add_Block(header)
        root_id  = header.ptr
        if root_id == 0:
            root_id = 1
            header.ptr = root_id
            root = LeafNode(root_id)
            header.block_sum += 1
            self.indexcache.add_Block(root)
            # Header.update_Header(header,self.index_file)
        else:
            root = self.indexcache.get_Block(root_id)
        self.insert(root,entry)

    def insert(self,cur_node,entry):
        """
        插入数据,初始传入根节点，从缓冲获取头结点
        :param entry:
        :return:
        """
        header = self.indexcache.get_Block(0)
        self.indexcache.add_Block(cur_node)
        if cur_node.is_leaf:
            cur_node.add_item(entry)
            # 更新头结点全局mbr
            header.mbr = header.update_mbr(entry[0])
            # 处理溢出
            if cur_node.is_overflow():
                self.handle_Overflow(cur_node)
            # 向上调整mbr
            self.adjust_Tree(cur_node)
        else:
            sub = self.choose_SubTree(cur_node,entry)
            self.insert(sub, entry)

    def adjust_Tree(self,cur_node):
        """
        向上调整MBR
        :param cur_node:
        :return:
        """
        global IO_COUNT
        if cur_node.parent_id != 0:  # 到根则停止
            par_id = cur_node.parent_id
            parent = self.indexcache.get_Block(par_id)
            if parent == None:
                parent = self.load_block(INDEX_FILE, par_id)
                IO_COUNT += 1
                self.indexcache.add_Block(parent)
            parent.update_Mbr()
            self.adjust_Tree(parent)

    def choose_SubTree(self, node, entry):
        """
        判断子节点是叶子还是分支节点，叶子则按最小重叠面积扩张
        分支则和R树相同
        :param node:
        :param entry:
        :return:
        """
        if node.is_leaf:
            return node
        else:
            global IO_COUNT
            child_id = node.item_list[0][1]
            child = self.indexcache.get_Block(child_id)  # 选择第一个子节点来判断
            if child == None:
                IO_COUNT += 1
                child = self.load_block(INDEX_FILE, child_id)
                self.indexcache.add_Block(child)
            if not child.is_leaf:  # 子节点为分支，则按最小扩展面积来选择子树
                min_area_en = float('inf')
                # area_enlarge_of_children = []
                sub = None
                for item in node.item_list:
                    area_enlarge = self.cal_area_enlargement(item[0], entry[0])
                    if area_enlarge == 0:  # 没扩张则直接选择
                        best_node = self.indexcache.get_Block(item[1])
                        if best_node == None:
                            best_node = self.load_block(INDEX_FILE, item[1])
                            IO_COUNT += 1
                            self.indexcache.add_Block(best_node)
                        return best_node
                    if area_enlarge < min_area_en:
                        min_area_en = area_enlarge
                        sub = item  # 迭代最小扩张面积节点
                bestnode = self.indexcache.get_Block(sub[1])
                if bestnode == None:
                    bestnode = self.load_block(INDEX_FILE, sub[1])
                    IO_COUNT += 1
                    self.indexcache.add_Block(bestnode)
                return bestnode

            else:  # 子节点为叶子，则按加入数据后节点重叠面积扩张最小来选择
                """
                p = 32
                计算节点的重叠面积之和
                """
                id_overlap_map = {}  # 保存扩张重叠面积和节点id映射
                ori_overlap_sum = self.cal_node_overlap(node)
                for item in node.item_list:  # 计算数据加入每个子节点后节点的重叠面积增加值
                    temp_id = item[1]
                    temp_mbr = self.get_mbr(item[0], entry[0])
                    overlap_sum = self.cal_node_overlap(node, temp_mbr, temp_id)
                    if overlap_sum == ori_overlap_sum:  # 重叠面积增加为0则直接选择
                        best_node = self.indexcache.get_Block(temp_id)
                        if best_node == None:
                            best_node = self.load_block(INDEX_FILE, temp_id)
                            IO_COUNT += 1
                        return best_node
                    id_overlap_map[temp_id] = overlap_sum - ori_overlap_sum
                # 排序字典,并选择使节点重叠面积增加最小的那个子节点id
                min_overlap_id = list(sorted(id_overlap_map.items(), key=lambda item: item[1]))[0][0]
                best_node = self.indexcache.get_Block(min_overlap_id)
                if best_node == None:
                    best_node = self.load_block(INDEX_FILE, min_overlap_id)
                    IO_COUNT += 1
                return best_node

    def get_mbr(self, mbr1, mbr2):
        """
        计算mbr2加入mbr1之后的大mbr，与节点更新mbr操作无关
        :param mbr1:
        :param mbr2:
        :return:
        """
        x1 = min(mbr1[0], mbr2[0])
        y1 = min(mbr1[1], mbr2[1])
        x2 = max(mbr1[2], mbr2[2])
        y2 = max(mbr1[3], mbr2[3])
        return [x1, y1, x2, y2]

    def cal_node_overlap(self, node, temp_mbr=None, temp_id=None):
        """
        计算节点的子节点mbr的全部重叠面积之和,temp_mbr是数据依次加入每个子节点mbr后的mbr
        注：不是子节点更新mbr操作
        :param node:
        :return:
        """
        overlap_sum = 0
        if temp_id == None and temp_mbr == None:
            for item1 in node.item_list:
                for item2 in node.item_list:
                    if item1[1] != item2[1]:
                        overlap_sum += self.cal_overlap_area(item2[0],item1[0])
                    else:
                        continue
            return overlap_sum
        else:
            for item1 in node.item_list:
                for item2 in node.item_list:
                    if item1[1] != item2[1] and item2[1] != temp_id:
                        overlap_sum += self.cal_overlap_area(item2[0], item1[0])
                    if item1[1] != item2[1] and item2[1] == temp_id:
                        overlap_sum += self.cal_overlap_area(temp_mbr, item1[0])
                    if item1[1] != item2[1] and item1[1] == temp_id:
                        overlap_sum += self.cal_overlap_area(item2[0], temp_mbr)
                    if item2[1] == item1[1]:
                        continue
            return overlap_sum

    def cal_overlap_area(self, mbr1, mbr2):
        """
        计算两个mbr的重叠面积
        :param mbr1:
        :param mbr2:
        :return:
        """
        if self.is_overlap(mbr1, mbr2):
            return min(abs(mbr1[2]-mbr2[0]),abs(mbr2[2]-mbr1[0])) *\
                min(abs(mbr1[3]-mbr2[1]),abs(mbr2[3]-mbr1[1]))
        return 0

    def is_overlap(self, mbr1, mbr2):
        """
        判断重叠
        :param mbr1:
        :param mbr2:
        :return:
        """
        return mbr1[0] <= mbr2[2] and mbr1[2] >= mbr2[0]\
            and mbr1[1] <= mbr2[3] and mbr1[3] >= mbr2[1]

    def handle_Overflow(self, cur_Node):
        """
        处理溢出，如果分裂传播到根，则新建一个根节点，然后调整对应的mbr
        :param cur_Node:
        :return:
        """
        cur_Node, new_Node = self.split_Node(cur_Node)
        header = self.indexcache.get_Block(0)
        global IO_COUNT
        if cur_Node.parent_id == 0:  # 分裂传播到根
            new_root = InterNode(header.block_sum)
            new_root.add_item([cur_Node.mbr,cur_Node.block_id])
            new_root.add_item([new_Node.mbr,new_Node.block_id])
            cur_Node.parent_id = new_root.block_id
            new_Node.parent_id = new_root.block_id
            self.indexcache.add_Block(new_root)
            self.indexcache.add_Block(cur_Node)
            self.indexcache.add_Block(new_Node)
            header.block_sum += 1
            header.ptr = new_root.block_id
        else:
            p_id = cur_Node.parent_id
            par = self.indexcache.get_Block(p_id)
            if par == None:
                par = self.load_block(INDEX_FILE, p_id)
                self.indexcache.add_Block(par)
            par.add_item([new_Node.mbr,new_Node.block_id])
            new_Node.parent_id = p_id
            self.indexcache.add_Block(cur_Node)
            self.indexcache.add_Block(new_Node)
            if par.is_overflow():
                self.handle_Overflow(par)
        self.adjust_Tree(cur_Node)
        self.adjust_Tree(new_Node)

    def split_Node(self, cur_Node):
        """
        分裂算法，令m等于0.45M，这样的话M-2m+2等于112-100+2=14种划分
        :param cur_Block:
        :return:
        """
        global IO_COUNT
        header = self.indexcache.get_Block(0)
        if cur_Node.is_leaf:
            new_Node = LeafNode(header.block_sum)
        else:
            new_Node = InterNode(header.block_sum)
        header.block_sum += 1
        self.indexcache.add_Block(new_Node)

        # m所占M的比率
        ratio = 0.4
        # 分裂节点当前数据项大小
        length = len(cur_Node.item_list)
        # 节点最大容量
        M = MAX_ITEM_AMOUNT
        m = int(ratio * M)

        # x轴对应的周长总和
        perimeter_xsum = 0

        # y轴对应的周长总和
        perimeter_ysum = 0

        # len(k)等于划分的数量
        k = list(range(1, M - 2 * m + 3))  # k= [1,2,...14]

        # 保存属于x轴的每个划分对应的重叠面积
        # 二维数组，内部子项为[[group1,group2],overlap_area]
        overlap_of_x_axis = []
        # 保存属于y轴的每个划分对应的重叠面积
        overlap_of_y_axis = []

        # 沿着x1排序mbr
        sorted_mbr_by_x1 = sorted(cur_Node.item_list, key=lambda x: x[0][0])
        for k_elem in k:  # 定义len(k)种划分
            for j in range(m - 1 + k_elem, length - (m - 1 + k_elem)):
                # 组1为前(m-1+k_elem)个项
                group1 = sorted_mbr_by_x1[:j]
                # 组2为剩余项目
                group2 = sorted_mbr_by_x1[j:]
                # 计算组1对应的周长
                group1_margin = self.cal_group_margin(group1)
                # 计算组2对应的周长
                group2_margin = self.cal_group_margin(group2)
                # 划分所对应的周长
                partition_margin = group2_margin + group1_margin
                # 划分对应的重叠面积
                partition_overlap_area = self.cal_partition_overlap(group1, group2)
                # 保存到x列表，以便后续选择最佳划分
                overlap_of_x_axis.append([[group1, group2], partition_overlap_area])
                # 迭代求和x轴对应的周长
                perimeter_xsum = perimeter_xsum + partition_margin

        # 沿着x2排序mbr
        sorted_mbr_by_x2 = sorted(cur_Node.item_list, key=lambda x: x[0][2])
        for k_elem in k:
            for j in range(m - 1 + k_elem, length - (m - 1 + k_elem)):
                group1 = sorted_mbr_by_x2[:j]
                group2 = sorted_mbr_by_x2[j:]
                group1_margin = self.cal_group_margin(group1)
                group2_margin = self.cal_group_margin(group2)
                partition_margin = group2_margin + group1_margin
                partition_overlap_area = self.cal_partition_overlap(group1, group2)
                overlap_of_x_axis.append([[group1, group2], partition_overlap_area])
                perimeter_xsum = perimeter_xsum + partition_margin

        # 沿着y1排序mbr
        sorted_mbr_by_y1 = sorted(cur_Node.item_list, key=lambda x: x[0][1])
        for k_elem in k:
            for j in range(m - 1 + k_elem, length - (m - 1 + k_elem)):
                group1 = sorted_mbr_by_y1[:j]
                group2 = sorted_mbr_by_y1[j:]
                group1_margin = self.cal_group_margin(group1)
                group2_margin = self.cal_group_margin(group2)
                partition_margin = group2_margin + group1_margin
                partition_overlap_area = self.cal_partition_overlap(group1, group2)
                overlap_of_y_axis.append([[group1, group2], partition_overlap_area])
                perimeter_ysum = perimeter_ysum + partition_margin

        # 沿着y2排序mbr
        sorted_mbr_by_y2 = sorted(cur_Node.item_list, key=lambda x: x[0][3])
        for k_elem in k:
            for j in range(m - 1 + k_elem, length - (m - 1 + k_elem)):
                group1 = sorted_mbr_by_y2[:j]
                group2 = sorted_mbr_by_y2[j:]
                group1_margin = self.cal_group_margin(group1)
                group2_margin = self.cal_group_margin(group2)
                partition_margin = group2_margin + group1_margin
                partition_overlap_area = self.cal_partition_overlap(group1, group2)
                overlap_of_y_axis.append([[group1, group2], partition_overlap_area])
                perimeter_ysum = perimeter_ysum + partition_margin

        """
        取x轴和y轴对应的周长和较小者,然后再在那个轴上选择重叠面积最小的最佳划分
        """
        # x轴对应的周长和更小
        if perimeter_xsum < perimeter_ysum:
            # 对x轴对应的划分按照重叠面积进行排序
            sorted_partition = sorted(overlap_of_x_axis, key=lambda x: x[1])
        else:  # y轴对应的周长和更小
            sorted_partition = sorted(overlap_of_y_axis, key=lambda x: x[1])
        # 选择重叠面积最小的那个划分
        best_partition = sorted_partition[0][0]
        group1 = best_partition[0]
        group2 = best_partition[1]

        # 将组1内的项目分配给分裂节点
        cur_Node.item_list = group1
        cur_Node.update_Mbr()
        # 将组2内的项目分配给新节点
        new_Node.item_list = group2
        new_Node.update_Mbr()

        # 临时存储叶子节点包含的数据块号
        dblock_id1 = []
        dblock_id2 = []

        # 更新叶子结点dblock_id列表
        if cur_Node.is_leaf:
            for mbr, dblock_id in cur_Node.item_list:
                dblock_id1.append(dblock_id)
            cur_Node.dblock_id_list = dblock_id1
            for mbr, dblock_id in new_Node.item_list:
                dblock_id2.append(dblock_id)
            new_Node.dblock_id_list = dblock_id2
        # 更新子节点的parent_id
        else:
            for item in cur_Node.item_list:
                child = self.indexcache.get_Block(item[1])
                if child == None:
                    child = self.load_block(self.index_file, item[1])
                    IO_COUNT += 1
                child.parent_id = cur_Node.block_id
            for item1 in new_Node.item_list:
                child1 = self.indexcache.get_Block(item1[1])
                if child1 == None:
                    child1 = self.load_block(self.index_file, item1[1])
                    IO_COUNT += 1
                child1.parent_id = new_Node.block_id

        return cur_Node, new_Node

    def cal_partition_overlap(self, group1, group2):
        """
        计算划分对应的重叠面积
        :param group:
        :return:
        """
        x1 = min([mbr[0] for mbr, block_id in group1])
        y1 = min([mbr[1] for mbr, block_id in group1])
        x2 = max([mbr[2] for mbr, block_id in group1])
        y2 = max([mbr[3] for mbr, block_id in group1])
        # 第一个组内的大mbr
        group1_mbr = [x1, y1, x2, y2]
        x3 = min([mbr[0] for mbr, block_id in group2])
        y3 = min([mbr[1] for mbr, block_id in group2])
        x4 = max([mbr[2] for mbr, block_id in group2])
        y4 = max([mbr[3] for mbr, block_id in group2])
        # 第二个组内的大mbr
        group2_mbr = [x3, y3, x4, y4]
        return self.cal_overlap_area(group1_mbr,group2_mbr)

    def cal_group_margin(self, group):
        """
        计算划分内的一组对应的周长，margin(group) = margin(bb(group))
        其中bb是组内的小矩形所组成的大矩形
        :param partion:
        :return:
        """
        x1 = min([mbr[0] for mbr,block_id in group])
        y1 = min([mbr[1] for mbr,block_id in group])
        x2 = max([mbr[2] for mbr,block_id in group])
        y2 = max([mbr[3] for mbr,block_id in group])
        # 组内的小mbr组成的大mbr
        group_mbr = [x1, y1, x2, y2]
        group_margin = 2 * (x2-x1) + 2 * (y2-y1)
        return group_margin

    def cal_area_enlargement(self, mbr1, mbr2):
        """
        计算mbr2，加入mbr1之后扩张的面积
        :param mbr1:
        :param mbr2:
        :return:
        """
        x1 = min(mbr1[0], mbr2[0])
        y1 = min(mbr1[1], mbr2[1])
        x2 = max(mbr1[2], mbr2[2])
        y2 = max(mbr1[3], mbr2[3])
        return (x2-x1)*(y2-y1) - self.cal_area(mbr1)

    def cal_area(self, mbr):
        """
        计算矩形面积
        :param mbr:
        :return:
        """
        return abs(mbr[2]-mbr[0]) * abs(mbr[3]-mbr[1])


class Data:
    """
    数据类，形成数据文件
    """
    def __init__(self, d_cache, index, d_file=D_FILE):
        self.d_file = d_file
        self.d_cache = d_cache
        self.index = index

    @staticmethod
    def load_dblock(dblock_id, datafile=D_FILE):
        """
        从数据文件读取一块数据块
        :param dblock_id:
        :param datafile: 数据文件
        :return:
        """
        if not os.path.exists(datafile):
            with open(datafile, 'wb') as f:
                pass
        with open(datafile, 'rb') as f:
            f.seek(dblock_id*BLOCK_SIZE)
            bindata = f.read(BLOCK_SIZE)
            return DataBlock.unpack(bindata)

    def insert_points(self, point):
        global DB_COUNT
        global point_count
        db = self.choose_dblock(point)
        print(f'插入第{point_count}个点')
        point_count += 1
        db.add_points(point)
        """
        数据块如果满了，则：
        1   将数据块对应的ID和mbr加入索引块
        2   创建一块新的数据块
        """
        if db.is_overflow():
            mbr = db.get_dblock_mbr()
            self.index.create_Index([mbr, db.block_id])
            self.create_new_block()

    def choose_dblock(self, point):
        """
        每次选择数据点距离数据块中心最近的那块
        :param point: 数据点
        :return:
        """
        global DB_COUNT
        global IO_COUNT

        # 创建初始块
        if DB_COUNT == 0:
            block = DataBlock(0)
            self.d_cache.add_dblock(block)
            DB_COUNT += 1
            return block
        # 直接选择块号最大的那一块，然后插入数据
        block = self.d_cache.get_dblock(DB_COUNT - 1)
        if block == None:
            block = self.load_dblock(DB_COUNT - 1)
            IO_COUNT += 1
        return block

    def create_new_block(self):
        """
        生成一块新的数据块并加入缓冲
        :param dblock:
        :return:
        """
        global DB_COUNT
        new_dblock = DataBlock(DB_COUNT)
        DB_COUNT += 1
        self.d_cache.add_dblock(new_dblock)
        return


class DataBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.data_points = []  # 存储Point对象

    def is_overflow(self):
        return (len(self.data_points) + 1) * 16 + 8 > BLOCK_SIZE

    def pack(self):
        binfo = struct.pack('2I', self.block_id, len(self.data_points))
        for x, y in self.data_points:
            binfo = binfo + struct.pack('2d', x, y)
        return binfo

    @staticmethod
    def unpack(bindata):
        block_id, length = struct.unpack('2I', bindata[:8])
        dblock = DataBlock(block_id)
        offset = 8
        for i in range(length):
            x = struct.unpack('d', bindata[offset:offset+8])[0]
            offset += 8
            y = struct.unpack('d', bindata[offset:offset+8])[0]
            offset += 8
            dblock.data_points.append([x, y])
        return dblock

    def add_points(self, point):
        self.data_points.append(point)

    def get_dblock_mbr(self):
        """
        计算数据块的所有数据点对应的mbr
        :return:
        """
        x1 = min([x for x, y in self.data_points])
        y1 = min([y for x, y in self.data_points])
        x2 = max([x for x, y in self.data_points])
        y2 = max([y for x, y in self.data_points])
        return [x1, y1, x2, y2]


class Cache:
    def __init__(self, max_amount=BUFFER_SIZE):
        self.id_block_map = {}
        # 字典不能索引，列表用来选择置换的块id
        self.block_id_list = []
        self.max_amount = max_amount
        pass

    def is_full(self):
        return len(self.id_block_map) >= self.max_amount


class InsertDataCache(Cache):
    def __init__(self):
        Cache.__init__(self)

    def add_dblock(self, b):
        if b.block_id not in self.id_block_map:
            self.id_block_map[b.block_id] = b
            self.block_id_list.append(b.block_id)
        if self.is_full():
            self.fifo()

    def get_dblock(self,block_id):
        for id in self.id_block_map:
            if id==block_id:
                return self.id_block_map[block_id]
        return None

    def write_dblock(self,dblock):
        if not os.path.exists(D_FILE):
            with open(D_FILE,'wb') as f:
                pass
        with open(D_FILE, 'rb+') as f:
            f.seek(dblock.block_id*BLOCK_SIZE)
            f.write(dblock.pack())

    def fifo(self):
        """
        先进先出置换策略，只要缓冲一满：
        1   将要置换的块写入文件
        2   将要置换的块从ID列表中舍去
        :return:
        """
        global IO_COUNT
        id = self.block_id_list[0]
        IO_COUNT += 1
        # 将数据块写入文件
        self.write_dblock(self.id_block_map[id])
        firstid = self.block_id_list.pop(0)
        del self.id_block_map[firstid]

    def update_to_disk(self):
        """
        索引建立完成后将缓冲中的内容全部写入磁盘
        :return:
        """
        if not os.path.exists(D_FILE):
            with open(D_FILE, 'wb') as f:
                pass
        global IO_COUNT
        with open(D_FILE, 'rb+') as fp:
            for block_id in self.block_id_list:
                IO_COUNT += 1
                self.write_dblock(self.id_block_map[block_id])
        self.id_block_map = {}
        self.block_id_list = []


class InsertIndexCache(Cache):
    def __init__(self):
        Cache.__init__(self)
        # self.block_modified_dict = {}#{id:is_modified}

    def write_block_to_file(self,block):
        with open(INDEX_FILE,'rb+') as  fp:
            fp.seek(block.block_id*BLOCK_SIZE)
            fp.write(block.pack())

    def add_Block(self,block) -> None:
        if block.block_id not in self.id_block_map:
            # self.id_block_map[block.block_id] = bytes(block.pack())
            self.id_block_map[block.block_id] = block
            self.block_id_list.append(block.block_id)
            # self.block_modified_dict[block.block_id] = True
        if self.is_full():
            self.fifo()

    def get_Block(self,block_id):
        """
        注：函数内的变量最好不要和形参名字一样，不然会出错，例如如果block_id1
        也写成block_id，则第一次就判断正确，返回的就是第一个加入缓冲区的
        :param block_id:
        :return:
        """
        for block_id1 in self.block_id_list:
            if block_id == block_id1:
                return self.id_block_map[block_id1]
        return None

    def fifo(self):
        global  IO_COUNT
        header = self.get_Block(0)
        root_id  = header.ptr
        for id in self.block_id_list:
            if id != 0 and id != root_id:#头和根常驻内存
                IO_COUNT += 1
                self.write_block_to_file(self.id_block_map[id])
                del self.id_block_map[id]
                self.block_id_list.remove(id)

    def update_to_disk(self):
        """
        将缓冲区中的内容写入磁盘
        :return:
        """
        if not os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'wb'):
                pass
        global IO_COUNT
        with open(INDEX_FILE,'rb+') as fp:
            for block_id in self.block_id_list:
                IO_COUNT += 1
                self.write_block_to_file(self.id_block_map[block_id])
        self.id_block_map = {}
        self.block_id_list = []


class QueryDataCache(Cache):
    def __init__(self):
        Cache.__init__(self)

    def add_dblock(self,b):
        if b.block_id not in self.id_block_map:
            self.id_block_map[b.block_id] = b
            self.block_id_list.append(b.block_id)
        if self.is_full():
            self.fifo()

    def get_dblock(self,block_id):
        for id in self.id_block_map:
            if id == block_id:
                return self.id_block_map[block_id]
        return None

    def fifo(self):
        """
        先进先出
        :return:
        """
        global IO_COUNT
        id = self.block_id_list[0]
        firstid = self.block_id_list.pop(0)
        del self.id_block_map[firstid]


class QueryIndexCache(Cache):
    def __init__(self):
        Cache.__init__(self)

    def add_Block(self,block) -> None:
        if block.block_id not in self.id_block_map:
            # self.id_block_map[block.block_id] = bytes(block.pack())
            self.id_block_map[block.block_id] = block
            self.block_id_list.append(block.block_id)
            # self.block_modified_dict[block.block_id] = True
        if self.is_full():
            self.fifo()

    def get_Block(self,block_id):
        for block_id1 in self.block_id_list:
            if block_id == block_id1:
                return self.id_block_map[block_id1]
        return None

    def fifo(self):
        header = self.get_Block(0)
        root_id  = header.ptr
        for id in self.block_id_list:
            if id != 0 and id != root_id:  # 头和根常驻内存
                del self.id_block_map[id]
                self.block_id_list.remove(id)


def main():
    print('1.插入')
    print('2.查询')
    choice = input('请输入选择:')

    if choice == '1':
        # 创建插入缓冲对象
        insert_indexcache = InsertIndexCache()
        insert_datacache = InsertDataCache()

        # 先排序数据再建立索引，否则很耗时,利用str算法排序
        datapoints = str_sort()

        # datapoints = get_data()
        # rank_space_points = to_rank_space(datapoints)

        # 利用希尔伯特曲线排序
        # hv_points_map = cal_hilbert_value_in_origin(datapoints)
        # sorted_map = dict(sorted(hv_points_map.items()))

        # 利用z曲线排序
        # zv_points_map = cal_z_values(rank_space_points, datapoints)
        # sorted_map = dict(sorted(zv_points_map.items()))

        # datapoints = []
        # for item in sorted_map:
        #     datapoints.append(sorted_map[item])
        # print(datapoints)

        index = Index(insert_indexcache,insert_datacache)
        data = Data(index=index, d_cache=insert_datacache)

        # 开始构建索引
        start_time = time.time()
        for dt in datapoints:
            data.insert_points(dt)

        # 索引构建完成，将缓冲的内容写回文件
        insert_indexcache.update_to_disk()
        insert_datacache.update_to_disk()
        print(f'建树io为{IO_COUNT}次')
        print(f'建树时间为{time.time() - start_time}秒')

    elif choice == '2':
        # 创建查询缓冲对象
        query_indexcache = QueryIndexCache()
        query_datacache = QueryDataCache()

        # 创建索引和查询对象
        index = Index(query_indexcache, query_datacache)
        query = Query(index, query_indexcache, query_datacache)

        # 读取查询数据
        with open(QUERY_FILE, 'r') as fp:
            data = fp.readlines()

        query_list = []  # 查询列表
        result = set()  # 结果集
        # result = []  # 结果集
        query_num = 0  # 查询数目

        # 将字符串转换成浮点数
        for line in data:
            quer = list(map(float, line.strip().split(',')))
            query_list.append(quer)

        # 开始计时
        start_time = time.time()
        for que in query_list:
            query_num += 1
            if len(que) == 4:  # 代表范围查询
                result = result.union(query.range_query(que))
                # result = result + (query.range_query(que))
            else:  # 点查询
                result = result.union(query.point_query(que))
                # result = result + (query.point_query(que))
        print(len(result))  # 查询结果数量
        print(f'平均查询IO次数为{IO_COUNT / query_num}')
        print(f'平均查询时间为{(time.time() - start_time) / query_num}秒')


if __name__ == '__main__':
    main()
