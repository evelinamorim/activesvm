import sys
import re
import json
from empath import Empath

NUM_PATTERN = re.compile('\d+')
REASON_ID_PATTERN = re.compile('\d+\_\d+')

support = ['because','therefore', 'after', 'for', 'since', 'when', 'assuming', 'so', 'accordingly', 'thus', 'hence', 'then', 'consequently']
conflict = ['however', 'but', 'though', 'except', 'not', 'never', 'no','whereas', 'nonetheless', 'yet', 'despite']
conclusion = ['as a result']
complementary = ['while','whereas', 'whereas normally','whereas otherwise','not even', 'yet']
causal_argument = ['cause', 'effect', 'means', 'end', 'makes that', 'leads to', 'cultivate', 'suddenly', 'in one blow', 'will yield', 'is', 'guarantee for', 'necessarily']

verbs_hedging = ['appear','assume', 'attempt','believe','calculate','estimate','imply','indicate','note','predict','propose', 'report','see','seek','seem','speculate','suggest', 'suspect']

def read_data(file_name):

    json_fd = open(file_name, "rb")
    data_list = {}
    avg_len = 0
    k = 0
    for line in json_fd:
        d = json.loads(line.decode('UTF-8'))
        data_list[d["commentID"]] = d["propositions"]
        avg_len = avg_len + len(d["propositions"])
        k = k + 1
    print("-->",avg_len/k)
    json_fd.close()

    return data_list

def get_connect_unconnect(arg_list):

    arg_dict = {}
    for a in arg_list:
       arg_dict[str(a["id"])] = (a["text"], a["reasons"])

    set_connected = [0]*len(arg_dict.keys())
    connected = {}

    for id_a in arg_dict:

        text_a, reasons_a = arg_dict[id_a]

        if reasons_a is not None:

            set_connected[int(id_a)] = 1
            connected[id_a] = []
            for r in reasons_a:
                if REASON_ID_PATTERN.match(r):
                    reasons = r.split("_")
                    for c in reasons:
                        set_connected[int(c)] = 1
                    connected[id_a].append(reasons)
                elif NUM_PATTERN.match(r):
                    connected[id_a].append([r])
                    set_connected[int(r)] = 1

    connected_list = []
    unconnected_list = []
    # import pdb
    # pdb.set_trace()
    for i in range(len(set_connected)):
        if set_connected[i] == 0:
            unconnected_list.append(arg_dict[str(i)][0])
        else:
            if str(i)  in connected:
                for connection in connected[str(i)]:
                    text_all = ""
                    for c in connection:
                        text_all = text_all + " " + arg_dict[c][0]
                    connected_list.append((arg_dict[str(i)][0], text_all))

    return connected_list, unconnected_list
 
def split_data(data_list):

    unconnected = []
    connected = []
    for d in data_list:
        c, u = get_connect_unconnect(data_list[d])
        unconnected.extend(u)
        connected.extend(c)

    return connected, unconnected

def count_connect(u):

    cat_all = {}
    lexicon = Empath()
    lexicon.create_category("support", support, model="nytimes")
    lexicon.create_category("conflict", conflict, model="nytimes")
    lexicon.create_category("conclusion", conclusion, model="nytimes")
    lexicon.create_category("complementary", complementary, model="nytimes")
    lexicon.create_category("causal_argument", causal_argument, model="nytimes")
    lexicon.create_category("verbs_hedging", verbs_hedging, model="nytimes")

    heads = []
    not_heads = []

    for (arg1, arg2) in u:
        heads.append(arg1)
        not_heads.append(arg2)

    norep_heads = list(set(heads))
    norep_not_heads = list(set(not_heads))
    args_conn = list(set(heads) | set(not_heads))

    lexicon = Empath()
    #cat_heads = lexicon.analyze(norep_heads, categories = ["support", "conflict", "conclusion", "complementary", "causal_argument"], normalize=True)
    cat_heads = lexicon.analyze(norep_heads, categories = ['verbs_hedging'], normalize=True)
    # cat_heads = {}
    # for h in norep_heads:
    #    cat_heads = lexicon.analyze(h, normalize=True)
    #    if cat_heads["fun"] != 0:
    #        print(h, cat_heads["fun"])
    # cat_not_heads = lexicon.analyze(norep_not_heads,categories = ["support", "conflict", "conclusion", "complementary", "causal_argument"], normalize=True)
    cat_not_heads = lexicon.analyze(norep_not_heads, categories = ['verbs_hedging'], normalize=True)
    # cat_all = lexicon.analyze(args_conn,categories = ["support", "conflict", "conclusion", "complementary", "causal_argument"], normalize=True)
    cat_all = lexicon.analyze(args_conn,  categories = ['verbs_hedging'],  normalize=True)

    return cat_heads, cat_not_heads, cat_all

def count_unconnect(u):

    # espero que seja um grupo bem diverso
    lexicon = Empath()
    # print(len(u))

    lexicon.create_category("support", support, model="nytimes")
    lexicon.create_category("conflict", conflict, model="nytimes")
    lexicon.create_category("conclusion", conclusion, model="nytimes")
    lexicon.create_category("complementary", complementary, model="nytimes")
    lexicon.create_category("causal_argument", causal_argument, model="nytimes")
    lexicon.create_category("verbs_hedging", verbs_hedging, model="nytimes")

#["because", "only", "before", "so", "if", "though", "then", "until", "once", "even", "since", "although", "so", "while", "having", "because", "already", "thus", "time", "unless", "now", "actually", "eventually"]
#["though", "although", "except", "yet", "but", "even", "because", "only", "Though", "Although", "Yet", "either", "nevertheless", "whereas", "though", "fact", "however", "unlike", "Furthermore", "because", "nonetheless", "And", "However", "none", "either", "still", "Even", "despite", "if", "so", "Yet", "meaning", "indeed", "consequently"]
#[]
#["while", "whereas", "though", "only", "yet", "While", "thus", "even", "Thus", "Instead", "although", "instead", "Though", "Moreover", "actually", "nevertheless", "sometimes", "still", "rather"]
#["means", "therefore", "means", "merely", "mechanism", "democratic_process", "Therefore", "simply", "free_market", "consequence", "because"]
    # cat_all = lexicon.analyze(u, categories = ["support", "conflict", "conclusion", "complementary", "causal_argument"], normalize=True)
    cat_all = lexicon.analyze(u, categories = ['verbs_hedging'], normalize=True)
    #cat_all = {}
    #for arg in u:
    #   cat = lexicon.analyze(arg)
    #   if cat["children"] != 0:
    #       print(arg, cat["children"])
    return cat_all

# TODO: intesercao entre unconnected arguments!
# TODO: intesercao entre connected arguments!

if __name__ == "__main__":
    data_list = read_data(sys.argv[1])
    c, u = split_data(data_list)
    cat_u = count_unconnect(u)
    print(cat_u)
    # rank_cat_u = list(cat_u.items())
    # rank_cat_u.sort(key=lambda x:x[1])
    # print(rank_cat_u[184:])
    print()

    h, nh, conn = count_connect(c)
    # rank_k = list(h.items())
    print(h)
    print()
    print(nh)
    print()
    print(conn)
    # rank_cat_h = list(h.items())
    # rank_cat_h.sort(key=lambda x:x[1])
    # fd = open("rancatuh.txt", "w")
    # for (cat, v) in rank_cat_h:
    #    min_v = min(v, cat_u[cat])
    #    if min_v != 0:
    #        ratio = max(v, cat_u[cat])/min(v, cat_u[cat])
    #    else:
    #        ratio = max(v, cat_u[cat])
    #    if ratio > 1.5:
    #        fd.write("%s,%.4f, %.4f,%f\n" % (cat,v, cat_u[cat],ratio))
    #fd.close()
    #print(rank_cat_nh)
    # print()
    # rank_cat_conn = list(conn.items())
    # rank_cat_conn.sort(key=lambda x:x[1])
    # print(rank_cat_conn[184:])

    # TODO: tranformar em lista de tuplas e ordenar
