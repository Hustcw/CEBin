import os
import sys
import binaryninja as bn
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method
import logging
import itertools
import time
import json

current_path = os.path.abspath(__file__)
current_path = '/'.join(current_path.split('/')[:-1]) + '/'
logging.basicConfig(filename=current_path+'../logs/prepare.log', level=logging.INFO)

# get mlil for each function
def get_mlil(bin, compiler, optimizer, pkg, arch, binary):
    bn.set_worker_thread_count(2)
    skip_function_list = [
        "_init", "_start", "_dl_relocate_static_pie", "deregister_tm_clones", "register_tm_clones", 
        "__do_global_dtors_aux", "frame_dummy", "__libc_csu_init", "__libc_csu_fini", "_fini"
    ]

    address_token_enums = [
        bn.InstructionTextTokenType.PossibleAddressToken,
        bn.InstructionTextTokenType.CodeRelativeAddressToken
    ]

    results = []
    with bn.open_view(bin, update_analysis=False) as bv:
        bv.update_analysis_and_wait()
        # iterate function symbols and get mlil
        for func_sym in bv.get_symbols_of_type(bn.SymbolType.FunctionSymbol):
            if func_sym.name in skip_function_list:
                continue
            # get function by address
            func = bv.get_function_at(func_sym.address)

            # define a class to store instruction length and tokens
            bb_cnt = 0
            instr_cnt = 0
            func_data = {}
            try:
                for block in func.medium_level_il:
                    for instr in block:
                        # print (hex(instr.address), instr.instr_index, instr.tokens)
                        symbolized_tokens = []
                        index_address = str(instr.instr_index)+'@'+hex(instr.address)
                        # replace address token with symbol name
                        for token in instr.tokens:
                            if token.type in address_token_enums:
                                symbol = bv.get_symbol_at(token.value)
                                if symbol is not None:
                                    symbolized_tokens.append(symbol.name.strip())
                                else:
                                    symbolized_tokens.append(token.text.strip())
                            else:
                                symbolized_tokens.append(token.text.strip())
                        
                        # merge index and address of each instruction: 2 @ 0x4005f0 -> 2@0x4005f0
                        tmp_replace_index = []
                        if len(symbolized_tokens) > 2:
                            i = 0
                            while i < len(symbolized_tokens) - 2:
                                if symbolized_tokens[i+1] == '@' and symbolized_tokens[i+2].startswith('0x') and symbolized_tokens[i].isdigit():
                                    symbolized_tokens = symbolized_tokens[:i] + [symbolized_tokens[i] + symbolized_tokens[i+1] + symbolized_tokens[i+2]] + symbolized_tokens[i+3:]
                                    tmp_replace_index.append(i)
                                i = i + 1                        
                        func_data[index_address] = symbolized_tokens
                        instr_cnt += 1  
                    bb_cnt += 1

                results.append((compiler, optimizer, pkg, arch, binary, func_sym.name, bb_cnt, instr_cnt, json.dumps(func_data)))
            except Exception as e:
                    logging.error("[ERROR] " + bin + ' ' + func_sym.name)
                    continue
    return results

def main(csv_path, out_dir):
    # Start the timer
    start_time = time.perf_counter()

    set_start_method("spawn")
    processes = 48
    logging.info("CPU count: " + str(processes) + " processes.")
    pool = Pool(processes=processes)

    df = pd.read_csv(csv_path)
    # dataframe groupby pakcage
    for pkg, df in df.groupby('package'):
        if os.path.exists(os.path.join(out_dir, pkg+".tsv")):
            logging.info(pkg + " already exists. Skip.")
            continue
        logging.info(pkg + " processing...")
        results = []
        for _, row in df.iterrows():
            _, compiler, optimizer, pkg, arch, binary, file_path, _ = row
            logging.info("Processing " + " ".join([compiler, optimizer, pkg, arch, binary, file_path]))
            results.append(pool.apply_async(get_mlil, (file_path, compiler, optimizer, pkg, arch, binary)))
        output = [p.get() for p in results]
        data = list(itertools.chain(*output))
        pkg_df = pd.DataFrame(data, columns=['compiler', 'optimizer', 'package', 'arch', 'binary', 'func_name', 'bb_cnt', 'instr_cnt', 'func_str'])
        pkg_df.to_csv(os.path.join(out_dir, pkg+".tsv"), index=False, sep='\t')
        logging.info(pkg + " done.")

    # End the timer
    end_time = time.perf_counter()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    logging.info(f'Elapsed time: {elapsed_time:.6f} seconds')

if __name__ == "__main__":
    main(current_path+"../data/raw/BinaryCorp/small_train.csv", current_path+"../data/processed/BinaryCorp/train")