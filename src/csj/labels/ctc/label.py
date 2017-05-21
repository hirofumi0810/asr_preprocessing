#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make ground truth labels for CTC network (CSJ corpus)."""

import os
import sys
import re
import numpy as np
import codecs
import MeCab
import jaconv
import time
from progressbar import ProgressBar
m = MeCab.Tagger("-Oyomi")

sys.path.append(os.pardir)
from prepare_path_dialogue import Prepare


def read(label_paths, save_path, label_type, social_signal_type, speaker):
    """Read diaogue text.
    Argss
        label_paths:
        save_path:
        label_type: katakana83 or katakana145 or phone
        social_signal_type: insert or replace or remove or ssonly
        speaker: left or right
    Returns:
        utterance_dict:
            key = > file_index
            value = > list of[start_time, end_time, utterance]
    """
    utterance_dict = {}

    print('Saving target labels...')
    p_read = ProgressBar(max_value=len(label_paths))
    i_path = 0
    for label_path in p_read(label_paths):
        # print(label_path)
        with codecs.open(label_path, 'r', 'shift-jis') as f:
            if speaker == 'left':
                file_name = os.path.basename(label_path).split('.')[0] + '-L'
            elif speaker == 'right':
                file_name = os.path.basename(label_path).split('.')[0] + '-R'
            utterances = []
            start_flag = True  # 最初の発話で前の発話時間を無視する用
            merge_flag = False  # サイレンスが短い場合に前後の発話をマージする用
            dont_merge_flag = False  # 前の発話が不適切な場合にマージを禁止する用
            end_pre = 0
            duration = 0
            duration_pre = 0
            label_save_name_pre = ''
            another_flag = False  # 相手の発話内容を無視する用

            for line in f:
                line = line.strip()

                # regular expression
                timestamp_left_expr = re.compile(
                    r'(\d{4})[\s]+(\d{5}).(\d{3})[-\s](\d{5}).(\d{3})[\s]L:(.*)')
                timestamp_right_expr = re.compile(
                    r'(\d{4})[\s]+(\d{5}).(\d{3})[-\s](\d{5}).(\d{3})[\s]R:(.*)')
                utterance_expr = re.compile(r'(.+)[\s]+\&\s(.+)')
                comment_expr = re.compile(r'\%(.*)')

                if speaker == 'left':
                    timestamp = re.search(timestamp_left_expr, line)
                    timestamp_another = re.search(timestamp_right_expr, line)
                elif speaker == 'right':
                    timestamp = re.search(timestamp_right_expr, line)
                    timestamp_another = re.search(timestamp_left_expr, line)
                utterance_raw = re.match(utterance_expr, line)
                comment = re.match(comment_expr, line)

                # skip header
                if start_flag and timestamp is None:
                    continue

                if timestamp is not None:
                    another_flag = False

                    if not start_flag:
                        if kakko != '':
                            yomi_stack = kakko[1:-1]
                            text_stack = ''

                        while '  ' in text_stack:
                            text_stack = text_stack.replace('  ', '')
                        filler_list = extract_filler(text_stack)
                        # social signal processing
                        utterance = regular_expression(
                            yomi_stack, social_signal_type, filler_list)

                        # convert from '__' to '_'
                        while '__' in utterance:
                            utterance = utterance.replace('__', '_')

                        if utterance != '_':
                            if merge_flag:
                                # merge with a previous utterance
                                # print('Merged...')
                                merge_flag = False
                                # update
                                utterance_info = utterances.pop()
                                label_save_name = utterance_info[
                                    0] + '_' + label_save_name.split('_')[1]
                                start = utterance_info[1]
                                utterance_info[2] = end
                                duration = end - start
                                utterance_info[3] += utterance
                                # print('\n' + utterance_info[3])
                            else:
                                # include silence at each side of the utterance
                                if utterance_info[1] - end_pre > 0.6:
                                    # reserve silence 0.5 sec between an
                                    # adjacent utterance
                                    if len(utterances) != 0:
                                        # previous utterance
                                        utterances[-1][2] = round(
                                            utterances[-1][2] + 0.5, 3)
                                    # current utterance
                                    utterance_info[1] = round(
                                        utterance_info[1] - 0.5, 3)
                                else:
                                    # if the interval < 0.5 sec, up to 0.1 sec
                                    if len(utterances) != 0:
                                        # previous utterance
                                        utterances[-1][2] = round(
                                            start - 0.1, 3)
                                    # current utterance
                                    utterance_info[1] = round(end_pre + 0.1, 3)

                                # save end time of current utterance for next
                                # one
                                end_pre = end
                                duration_pre = duration
                                label_save_name_pre = label_save_name
                                utterance_info.extend([utterance])
                            utterances.append(utterance_info)

                    ####################
                    # checking
                    ####################
                    if duration < 0:
                        print('Duration < 0.')
                        print(utterance_info)
                        sys.exit(0)
                    if duration > 50:
                        print('Duration too long.')
                        print(utterance_info)
                        sys.exit(0)

                    # initialization
                    start_sec = float(timestamp.group(2))
                    start_msec = float(timestamp.group(3)) * 0.001
                    start = start_sec + start_msec
                    end_sec = float(timestamp.group(4))
                    end_msec = float(timestamp.group(5)) * 0.001
                    end = end_sec + end_msec
                    duration = end - start
                    utt_index = timestamp.group(1)
                    label_save_name = file_name + '_' + utt_index
                    utterance_info = [label_save_name, start, end]
                    kakko = timestamp.group(6)
                    text_stack = ''
                    yomi_stack = '_'

                    if start_flag:
                        start_flag = False
                    # merge with the previous utterance, if the interval < 0.2
                    # sec
                    elif start - end_pre < 0.2:
                        merge_flag = True

                elif timestamp_another is not None:
                    # ignore listener's utterance
                    another_flag = True

                elif utterance_raw is not None:
                    text = utterance_raw.group(1)
                    yomi = utterance_raw.group(2)

                    # stack each line until next time stamp exists
                    if not another_flag:
                        text_stack += text
                        yomi_stack += yomi + '_'

                elif comment is not None:
                    # ignore comment
                    continue

                else:
                    pass

            ####################
            # last utterance
            ####################
            if kakko != '':
                yomi_stack = kakko[1:-1]
                text_stack = ''

            while '  ' in text_stack:
                text_stack = text_stack.replace('  ', '')
            filler_list = extract_filler(text_stack)
            # social signal processing
            utterance = regular_expression(
                yomi_stack, social_signal_type, filler_list)

            # convert from '__' to '_'
            while '__' in utterance:
                utterance = utterance.replace('__', '_')

            if utterance != '_':
                if merge_flag:
                    # merge with a previous utterance
                    # print('Merged...')
                    # update
                    utterance_info = utterances.pop()
                    label_save_name = utterance_info[0] + \
                        '_' + label_save_name.split('_')[1]
                    start = utterance_info[1]
                    utterance_info[2] = end
                    duration = end - start
                    utterance_info[3] += utterance
                    # print('\n' + utterance_info[3])
                else:
                    utterance_info.extend([utterance])
                utterances.append(utterance_info)

            ####################
            # checking
            ####################
            if duration < 0:
                print('Duration < 0.')
                print(utterance_info)
                sys.exit(0)
            if duration > 50:
                print('Duration too long.')
                print(utterance_info)
                sys.exit(0)

            # save all utterances in each file
            utterances_updated = []
            for each_utt in utterances:
                # convert from '__' to '_'
                # utt_text = str(each_utt[3])
                # while '__' in utt_text:
                #     utt_text = utt_text.replace('__', '_')
                # each_utt[3] = list(utt_text)

                if len(each_utt[3]) <= 2:
                    print(each_utt)

                # convert from string to list of number
                if label_type in ['katakana83', 'katakana145']:
                    kana_list = katakana2num(
                        each_utt[3], label_type, social_signal_type)
                elif label_type == 'phone':
                    phone_list = katakana2phone(each_utt[3])
                    kana_list = phone2num(phone_list)

                # convert social signals to number
                ss_dict = {'_': 0, 'L': 1, 'F': 2, 'B': 3, 'D': 4}
                laughter_flag = False
                for i in range(len(kana_list)):
                    if social_signal_type == 'ssonly':
                        if kana_list[i] in ['_', 'F', 'B', 'D']:
                            kana_list[i] = ss_dict[kana_list[i]]
                        elif kana_list[i] == 'L':
                            laughter_flag = True
                            kana_list[i] = 1
                        elif kana_list[i] == 'l':
                            laughter_flag = False
                            kana_list[i] = ''
                        else:
                            if laughter_flag:
                                kana_list[i] = 1
                            else:
                                # replace katakana to single 'speech' label
                                kana_list[i] = 5
                    else:
                        if kana_list[i] in ss_dict.keys():
                            kana_list[i] = ss_dict[kana_list[i]]

                    # checking
                    if (not isinstance(kana_list[i], int)) and kana_list[i] != '':
                        print('Label is not type int.')
                        print(kana_list)
                        print(each_utt)
                        sys.exit(0)

                while '' in kana_list:
                    kana_list.remove('')

                # save as npy file (only duration > 50msec)
                if each_utt[2] - each_utt[1] > 0.05:
                    # np.save(os.path.join(save_path, each_utt[0] + '.npy'), kana_list)
                    utterances_updated.append(each_utt)
                else:
                    print('Duration too short.')
                    print(each_utt)
                    sys.exit(0)

        utterance_dict[file_name] = utterances_updated
        p_read.update(i_path + 1)
        i_path += 1
        time.sleep(0.01)

    return utterance_dict


def regular_expression(utterance, social_signal_type, filler_list=None):
    """Process hierarchical structure.
    Args:
        utterance: hierarchical labeled text
        social_signal_type: insert or replace or remove or ssonly
        filler_list: list of fillers
    Returns:
        char_stack: text removed hierarchical structure
    """
    if utterance == '笑':
        return '_L_'
    elif utterance in ['雑音', '息', '咳', 'フロア発話']:
        return ''
    elif 'R' in utterance or '×' in utterance:
        return ''

    while '<笑>' in utterance:
        utterance = utterance.replace('<笑>', '_L_')
    while '<Q>' in utterance:
        utterance = utterance.replace('<Q>', '_')
    while '<H>' in utterance:
        utterance = utterance.replace('<H>', '')
    while '<FV>' in utterance:
        utterance = utterance.replace('<FV>', '')
    while '<息>' in utterance:
        utterance = utterance.replace('<息>', '')

    stack = []
    char_stack = ''
    for char in utterance:
        if char in ['(', '<']:
            stack.append(char_stack)
            char_stack = ''
        elif char in [')', '>']:
            # process in each ()
            char_stack, filler_list = make_social_singal_label(char_stack, social_signal_type,
                                                               filler_list)
            try:
                char_stack = stack.pop() + char_stack
            except:
                pass
        else:
            char_stack += char

    # 括弧閉じ忘れチェック
    if len(stack) != 0:
        char_stack, _ = make_social_singal_label(
            char_stack, social_signal_type, filler_list)
        char_stack = stack.pop() + char_stack

    # convert to katakana
    char_stack = m.parse(char_stack).strip()
    char_stack = jaconv.hira2kata(char_stack)

    # test
    # char_stack = char_stack.replace('L', '')
    # char_stack = char_stack.replace('F', '')
    # char_stack = char_stack.replace('?', '')
    # char_stack = char_stack.replace('D', '')
    # char_stack = char_stack.replace(' ', '')
    # if not is_katakana(char_stack):
    #     print(m.parse(char_stack).strip())
    #     print(utterance)
    #     # sys.exit(0)

    return char_stack


def extract_filler(utterance):

    if 'F' not in utterance:
        return []

    while '<FV>' in utterance or '<VN>' in utterance:
        utterance = utterance.replace('<FV>', '').replace('<VN>', '')

    stack = []
    char_stack = ''
    filler_flag = False
    filler_list = []

    for char in utterance:
        if char == 'F':
            filler_flag = True
        elif char == ')':
            filler_flag = False
            filler_list.append(char_stack)
            char_stack = ''

        if filler_flag:
            char_stack += char

    while '' in filler_list:
        filler_list.remove('')
    for i in range(len(filler_list)):
        filler_list[i] = filler_list[i][2:]

    if utterance.count('F') != len(filler_list):
        print(utterance)
        print(filler_list)

    return filler_list


def make_social_singal_label(text, social_signal_type, filler_list=None):
    """Process social signal labels.
    Args:
        text: text to read
        social_signal_type: insert or replace or remove or ssonly
        filler_list: list of fillers
    Returns:
        text_converted: text whose tag was processed
        filler_list: list of fillers(popped if used)
    """
    # regular expression
    # hierarchical
    M_expr = re.compile(r'M[\s](.+)')
    O_expr = re.compile(r'O[\s](.+)')
    which_expr = re.compile(r'W[\s](.*);(.*)')
    laughing_expr = re.compile(r'[L|笑][\s]([^(]*)', re.IGNORECASE)
    filler_expr = re.compile(r'F[\s]([^(]+)', re.IGNORECASE)
    disfluency_expr = re.compile(r'D[2]*[\s]([^(]+)', re.IGNORECASE)
    question_expr = re.compile(r'[\?][\s]([^(]+)', re.IGNORECASE)
    question_only_expr = re.compile(r'\?')
    # not hierarchical
    pause_expr = re.compile(r'P:\d{5}.\d{3}-\d{5}.\d{3}', re.IGNORECASE)
    B_expr = re.compile(r'B[\s](.*);(.*)')

    # search each labels
    M = re.match(M_expr, text)
    O = re.match(O_expr, text)
    which = re.match(which_expr, text)
    laughing = re.match(laughing_expr, text)
    filler = re.match(filler_expr, text)
    question = re.match(question_expr, text)
    question_only = re.match(question_only_expr, text)
    pause = re.match(pause_expr, text)
    disfluency = re.match(disfluency_expr, text)
    B = re.match(B_expr, text)

    ##############################
    # not hierarchical structure
    ##############################
    if pause is not None:
        # print(text)
        return '_', filler_list

    # Bって何？
    # elif B is not None:
    #     print(text)
    #     if B.group(2) != '':
    #         return B.group(2), filler_list
    #     elif B.group(1) != '':
    #         return B.group(1), filler_list
    #     else:
    #         return '', filler_list

    ##############################
    # hierarchical structure
    # 下にあるものほど階層が深い
    ##############################
    elif M is not None:
        return M.group(1), filler_list

    elif O is not None:
        return O.group(1), filler_list

    # 読みの候補が2つあるやつは後半を選択
    elif which is not None:
        if which.group(2) != '':
            return which.group(2), filler_list
        elif which.group(1) != '':
            return which.group(1), filler_list
        else:
            return '', filler_list

    elif laughing is not None:
        # print(text)
        # print(laughing.group(1))
        laughing_word = laughing.group(1)
        if social_signal_type in ['insert', 'replace']:
            return '_L' + laughing_word + '_', filler_list
        elif social_signal_type == 'remove':
            # social signalsを考慮しない場合は _ を挿入しない
            return laughing_word, filler_list
        elif social_signal_type == 'ssonly':
            return '_L' + laughing_word + 'l_', filler_list

    elif filler is not None:
        # print(text)
        # print(filler.group(1))
        filler_word = filler.group(1)
        if social_signal_type in ['replace', 'ssonly']:
            return '_F_', filler_list

        elif social_signal_type == 'insert':
            if filler_word == 'VN':
                filler_word = filler_list.pop(0)
            else:
                filler_list.pop(0)
            return '_F' + filler_word + '_', filler_list

        elif social_signal_type == 'remove':
            if filler_word == 'VN':
                filler_word = filler_list.pop(0)
            else:
                filler_list.pop(0)
            return '_' + filler_word + '_', filler_list

    elif disfluency is not None:
        # print(text)
        # print(disfluency.group(1))
        disfluency_word = disfluency.group(1)
        if social_signal_type in ['insert', 'replace', 'ssonly']:
            return '_D' + disfluency_word + '_', filler_list
        elif social_signal_type == 'remove':
            return '_' + disfluency_word + '_', filler_list

    elif question is not None:
        # print(text)
        # print(question.group(1))
        question_word = question.group(1)
        return question_word, filler_list

    elif question_only is not None:
        # print(text)
        return '', filler_list

    else:
        print(text)
        return text, filler_list


def is_katakana(text):
    a = [ch for ch in text if ("ア" <= ch <= "ン") or ch == "ー"]
    if len(text) == len(a):
        return True
    return False


def katakana2num(utterance, label_type, social_signal_type):
    """ convert from katakana to number
    Args:
        utterance: text
        label_type: katakana83 or katakana145 or phone
        social_signal_type: insert or replace or remove or ssonly
    Returns:
        num_list: list of katakana(int)
    """
    # read a mapping file
    kana_dict = {}
    mapping_file_name = 'kana2num_83.txt' if label_type == 'katakana83' else 'kana2num_145.txt'
    with open(os.path.join(os.path.abspath('../../erato/'), mapping_file_name)) as f:
        for line in f:
            line = line.strip()
            kana_num = re.search(r'([^\s]+)[\s]+([\d]+)', line)
            kana_dict[kana_num.group(1)] = int(kana_num.group(2))

    # convert from katakana to number (excepting social signals & silence)
    kana_list = list(utterance)
    if label_type == 'katakana83':
        for i in range(len(kana_list)):
            if social_signal_type == 'remove':
                if kana_list[i] in kana_dict.keys():
                    kana_list[i] = int(kana_dict[kana_list[i]]) - 4
            else:
                if kana_list[i] in kana_dict.keys():
                    kana_list[i] = int(kana_dict[kana_list[i]])

    elif label_type == 'katakana145':
        for i in range(len(kana_list)):
            # 促音が次に来ているか確認
            if i != len(kana_list) - 1:
                if kana_list[i] + kana_list[i + 1] in kana_dict.keys():
                    kana_list[i] = kana_list[i] + kana_list[i + 1]
                    kana_list[i + 1] = 'null'

            if social_signal_type == 'remove':
                if kana_list[i] in kana_dict.keys():
                    kana_list[i] = int(kana_dict[kana_list[i]]) - 4
            else:
                if kana_list[i] in kana_dict.keys():
                    kana_list[i] = int(kana_dict[kana_list[i]])

        while 'null' in kana_list:
            kana_list.remove('null')

    return kana_list


if __name__ == '__main__':
    prep = Prepare()
    label_paths = prep.label()

    read(label_paths=label_paths, save_path='', label_type='katakana145',
         social_signal_type='insert', speaker='right')
