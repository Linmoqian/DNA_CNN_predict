. 使用GM12878细胞系，基于在hg19坐标系当中处理后的数据集，其中包含DNA序列信息，mRNA半衰期特征等。

数据集文件包含训练集（train.h5）, 验证集（ valid.h5）, 测试集（test.h5） 三份文件。每份文件为h5格式，每份文件包含gene_id（基因id）, halflife （半衰期）, promoter（启动子）,label（标签） 四个键值。其含义如下：



gene_id (基因id)：数据的每条数据为通用的ENSID ，为Ensembl数据库使用的ID标识符，用于唯一标识不同的分子特征，例如基因、转录本、外显子和蛋白质等。

halflife（半衰期）：数据的每条数据以UTR5LEN、CDSLEN、INTRONLEN、UTR3LEN、UTR5GC、CDSGC、UTR3GC、ORFEXONDENSITY 顺序排列，每条数据经过标准化处理。半衰期具体数据含义如下：



promoter (启动子)每条数据为DNA序列数据，每一条数据为长度20000bp的DNA序列（TSS ±10000bp），其中的每一个碱基（ATCG）为one-hot 处理后的结果。碱基one-hot对应编号为{'A':0, 'C':1, 'G':2, 'T':3}。

label (标签)为预测目标，包含0，1两类数值，其中0表示该条数据对应的基因为低表达基因，1表示该条数据对应的基因为高表达基因。