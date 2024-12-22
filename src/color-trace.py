#!/usr/bin/env python
"""trace multiple color images with potrace"""

# color_trace_multi
# Written by ukurereh
# May 20, 2012

# 赵豪杰在 Python3.8 下重写
# 2021年8月5日

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


# 外部程序的路径
pngquant_command = 'pngquant'
pngnq_path = 'pngnq'
ImageMagick_convert_command = 'magick convert'
ImageMagick_identify_command = 'magick identify'
potrace_command = 'potrace'
potrace_options = ''

max_command_length = 1900  # 命令行长度限制
log_level = 1  # 不止是一个常数，它也会爱 -v/--verbose 选项影响

version = '1.01'

import os, sys
import shutil
import subprocess
import argparse
from glob import iglob
import functools
import multiprocessing
import queue
import tempfile
import time
import shlex
import re
from pprint import pprint


from svg_stack import svg_stack


def Print(*args, level=1):
    global log_level
    if log_level >= level:
        print(*args)


def process_command(command, stdinput=None, stdout_=False, stderr_=False):
    """在后台 shell 中运行命令，返回 stdout 和/或 stderr

    返回 stdout, stderr 或一个数组（stdout, stderr），取决于 stdout, stderr 参数
    是否为 True。如果遇到错误，则抛出。

    命令: 要运行的命令
    stdinput: data (bytes) to send to command's stdin, or None
    stdout_: True to receive command's stdout in the return value
    stderr_: True to receive command's stderr in the return value
"""
    stdin_pipe = (subprocess.PIPE if stdinput is not None else None)
    stdout_pipe = (subprocess.PIPE if stdout_ is True else None)
    stderr_pipe = subprocess.PIPE
   
    Print(f'命令：{command}')
    # macOS中使用shell=True有问题：命令不能正常执行  
    # process = subprocess.Popen(shlex.split(command),
    #                       stdin=stdin_pipe,
    #                       stderr=stderr_pipe,
    #                       stdout=stdout_pipe,
    #                       shell=True)
    process = subprocess.Popen(shlex.split(command),
                          stdin=stdin_pipe,
                          stderr=stderr_pipe,
                          stdout=stdout_pipe)

    stdoutput, stderror = process.communicate(input=stdinput)

    return_code = process.wait()
    if return_code != 0:
        raise Exception(stderror.decode(encoding=sys.getfilesystemencoding()))

    if stdout_ and not stderr_:
        return stdoutput
    elif stderr_ and not stdout_:
        return stderror
    elif stdout_ and stderr_:
        return (stdoutput, stderror)
    elif not stdout_ and not stderr_:
        return None

def rescale(source, target, scale, filter='lanczos'):
    """使用 ImageMagick 将图片重新缩放、转为 png 格式
"""
    if scale == 1.0:  # 不缩放。检查格式
        if os.path.splitext(source)[1].lower() not in ['.png']: # 非 png 则转格式
            command = f'{ImageMagick_convert_command} "{source}" "{target}"'
            process_command(command)
        else: # png 格式则直接复制
            shutil.copyfile(source, target)
    else:
        command = '{convert} "{src}" -filter {filter} -resize {resize}% "{dest}"'.format(
            convert=ImageMagick_convert_command, src=source, filter=filter, resize=scale * 100,
            dest=target)
        process_command(command)
    
    if os.path.exists(target):
        print(f"重缩放目标文件已写入: {target}")
    else:
        print(f"重缩放目标文件不存在: {target}")

def quantize_reduce_colors(source, quantized_target, num_colors, algorithm='mc', dither=None):
    """将源图像量化到指定数量的颜色，保存到量化目标

    量化：缩减颜色数量，只保留最主要的颜色

    使用指定的算法来量化图像。
    源：源图像的路径，必须是 png 文件
    量化目标：输出图像的路径
    颜色数：要缩减到的颜色数量，0 就是不量化
    算法：
        - 'mc' = median-cut 中切 (默认值, 只有少量颜色, 使用 pngquant)
        - 'as' = adaptive spatial subdivision 自适应空间细分 (使用 imagemagick, 产生的颜色更少)
        - 'nq' = neuquant (生成许多颜色, 使用 pngnq)
    拟色: 量化时使用的抖动拟色算法
        None: 默认，不拟色
        'floydsteinberg': 当使用 'mc', 'as', 和 'nq' 时可用
        'riemersma': 只有使用 'as' 时可用
    """
    # 创建和执行量化图像的命令

    if num_colors in [0, 1]:
        # 跳过量化，直接复制输入到输出
        shutil.copyfile(source, quantized_target)

    elif algorithm == 'mc':  # median-cut 中切
        if dither is None:
            dither_option = '--nofs'
        elif dither == 'floydsteinberg':
            dither_option = ''
        else:
            raise ValueError("对 'mc' 量化方法使用了错误的拟色类型：'{0}' ".format(dither))
       
        if not os.path.exists(source):
            raise FileNotFoundError(f"源文件不存在: {source}")
        if not os.path.exists(quantized_target):
            raise FileNotFoundError(f"量化目标文件不存在: {quantized_target}")
        
        command = f'{pngquant_command} --force {dither_option} --output "{quantized_target}" {num_colors} -- "{source}"'
        process_command(command)

    elif algorithm == 'as':  # adaptive spatial subdivision 自适应空间细分
        if dither is None:
            dither_option = 'None'
        elif dither in ('floydsteinberg', 'riemersma'):
            dither_option = dither
        else:
            raise ValueError("Invalid dither type '{0}' for 'as' quantization".format(dither))
        command = '{convert} "{src}" -dither {dither} -colors {colors} "{dest}"'.format(
            convert=ImageMagick_convert_command, src=source, dither=dither_option, colors=num_colors, dest=quantized_target)
        process_command(command)

    elif algorithm == 'nq':  # neuquant
        ext = "~quant.png"
        destdir = os.path.dirname(quantized_target)
        if dither is None:
            dither_option = ''
        elif dither == 'floydsteinberg':
            dither_option = '-Q f '
        else:
            raise ValueError("Invalid dither type '{0}' for 'nq' quantization".format(dither))
        command = '"{pngnq}" -f {dither}-d "{destdir}" -n {colors} -e {ext} "{src}"'.format(
            pngnq=pngnq_path, dither=dither_option, destdir=destdir, colors=num_colors, ext=ext, src=source)
        process_command(command)
        # 因为 pngnq 不支持保存到自定义目录，所以先输出文件到当前目录，再移动到量化目标
        old_output = os.path.join(destdir, os.path.splitext(os.path.basename(source))[0] + ext)
        os.rename(old_output, quantized_target)
    else:
        # 在错误到达这里前 argparse 应该已经先捕捉到了
        raise NotImplementedError('未知的量化算法 "{0}"'.format(algorithm))

def remap_with_palette(source, remapped_target, palette_image, dither=None):
    """用调色板图像的颜色重映射源图像，保存到重映射目标

    源: 源图像路径
    重映射目标: 输出保存路径
    调色板图像: 一个图像路径，它包含了 src 将重映射的颜色
    拟色: 重映射时的拟色算法
        选项有：None, 'floydsteinberg', 和 'riemersma'
"""

    if not os.path.exists(palette_image):  # 确认下调色板图像存在
        raise IOError("未找到重映射调色板：{0} ".format(palette_image))

    if dither is None:
        dither_option = 'None'
    elif dither in ('floydsteinberg', 'riemersma'):
        dither_option = dither
    else:
        raise ValueError("不合理的重映射拟色类型：'{0}' ".format(dither))

    # magick convert "src.png" -dither None -remap "platte.png" "output.png"
    command = f'{ImageMagick_convert_command} "{source}" -dither {dither_option} -remap "{palette_image}" "{remapped_target}"'
    process_command(command)

def make_color_table(source_image):
    """从源图像得到特征色，返回 #rrggbb 16进制颜色"""

    command = f'{ImageMagick_convert_command} "{source_image}"  -unique-colors txt:-'
    stdoutput = process_command(command, stdout_=True) # 这个输出中包含了颜色

    regex_pattern = '#[0-9A-F]{6}'
    IM_output = stdoutput.decode(sys.getfilesystemencoding())
    hex_colors = re.findall(regex_pattern, IM_output)

    return hex_colors

def get_nonpalette_color(palette, start_from_black=True, avoid_colors=None):
    """return a color hex string not listed in palette
    返回一个不在调色板内的16进制颜色字符串

    从黑色开始: 从黑色开始搜索颜色，否则从白色开始
    规避颜色: 一个列表, 指定在搜索时需要规避的颜色
"""
    if avoid_colors is None:
        final_palette = tuple(palette)
    else:
        final_palette = tuple(palette) + tuple(avoid_colors)
    if start_from_black:
        color_range = range(int('ffffff', 16))
    else:
        color_range = range(int('ffffff', 16), 0, -1)
    for i in color_range:
        color = "#{0:06x}".format(i)
        if color not in final_palette:
            return color
    # 当调色板加上规避颜色，包含所有颜色 #000000-#ffffff 时，抛出错误
    raise Exception("未能找到调色板之外的颜色")

def isolate_color(source, temp_target, target_layer, target_color, palette, stack=False):  # new version
    """将指定颜色区域替换为黑色，其他区域为白色

    源: 源图像路径，必须匹配调色板的颜色
    目标图层: 输出图像的路径
    目标颜色: 要孤立的颜色 (来自调色板)
    调色板: 包含例如 "#010101" 的列表. (从制作调色板输出得到)
    stack: 如果 True，在颜色索引之前的颜色为白，之后的为黑
"""
    color_index = palette.index(target_color)

    # 为了避免调色板包含纯黑和纯白，背景和前景色都是非调色板的颜色（黑或白）
    background_white = "#FFFFFF"
    foreground_black = "#000000"
    background_near_white = get_nonpalette_color(palette, False, (background_white, foreground_black))
    foreground_near_black = get_nonpalette_color(palette, True, (background_near_white, background_white, foreground_black))

    # 打开管道 stdin/stdout
    with open(source, 'rb') as source_file:
        stdinput = source_file.read()

    # 新建一个很长的命令，当它达到足够长度时就执行
    # 因为分别执行填充命令非常的慢
    last_iteration = len(palette) - 1  # new
    command_prefix = '{convert} "{src}" '.format(convert=ImageMagick_convert_command, src=source)
    command_suffix = ' "{target}"'.format(target=temp_target)
    command_middle = ''

    for i, color in enumerate(palette):
        # fill this color with background or foreground?
        if i == color_index:
            fill_color = foreground_near_black
        elif i > color_index and stack:
            fill_color = foreground_near_black
        else:
            fill_color = background_near_white

        command_middle += ' -fill "{fill}" -opaque "{color}"'.format(fill=fill_color, color=color)
        if len(command_middle) >= max_command_length or (i == last_iteration and command_middle):
            command = command_prefix + command_middle + command_suffix

            stdoutput = process_command(command, stdinput=stdinput, stdout_=True)
            stdinput = stdoutput
            command_middle = ''  # reset

    # 现在将前景变黑，背景变白
    command = '{convert} "{src}" -fill "{fillbg}" -opaque "{colorbg}" -fill "{fillfg}" -opaque "{colorfg}" "{dest}"'.format(
        convert=ImageMagick_convert_command, src=temp_target, fillbg=background_white, colorbg=background_near_white,
        fillfg=foreground_black, colorfg=foreground_near_black, dest=target_layer)
    process_command(command, stdinput=stdinput)

def fill_with_color(source, target):
    command = '{convert} "{src}" -fill "{color}" +opaque none "{dest}"'.format(
        convert=ImageMagick_convert_command, src=source, color="#000000", dest=target)
    process_command(command)

def get_width(source):
    """返回头像宽多少像素"""
    command = '{identify} -ping -format "%w" "{src}"'.format(
        identify=ImageMagick_identify_command, src=source)
    stdoutput = process_command(command, stdout_=True)
    width = int(stdoutput)
    return width

def trace(source, trace_target, output_color, suppress_speckles=2, smooth_corners=1.0, optimize_paths=0.2, width=None, height=None, resolution=None):
    """在指定的颜色、选项下，运行 potrace

    源: 源文件
    描摹目标: 输出目标文件
    输出颜色: 描摹路径填充的颜色
    抑制斑点像素数: 抑制指定像素数量的斑点
        (等同于 potrace --turdsize)
    平滑转角: 平滑转角: 0 表示不平滑, 1.334 为最大
        (等同于 potrace --alphamax)
    优化路径: 贝塞尔曲线优化: 0 最小, 5 最大
        (等同于 potrace --opttolerance)
    宽度: 输出的 svg 像素宽度, 默认 None. 保持原始比例.
"""

    width_param = f'--width {width}' if width is not None else ''
    height_param = f'--height {height}' if height is not None else ''
    resolution_param = f'--resolution {resolution}' if resolution is not None else ''

    command = f'''{potrace_command} --svg -o "{trace_target}" -C "{output_color}" -t {suppress_speckles} -a {smooth_corners} -O {optimize_paths} 
                {width_param} {height_param} {resolution_param} "{source}"'''
    Print(command)

    process_command(command)

def check_range(min, max, typefunc, typename, strval):
    """对 argparse 的参数，检查参数是否符合范围

    min: 可接受的最小值
    max: 可接受的最大值
    typefunc: 值转换函数, e.g. float, int
    typename: 值的类型, e.g. "an integer"
    strval: 包含期待值的字符串
"""
    try:
        val = typefunc(strval)
    except ValueError:
        msg = "must be {typename}".format(typename=typename)
        raise argparse.ArgumentTypeError(msg)
    if (max is not None) and (not min <= val <= max):
        msg = "must be between {min} and {max}".format(min=min, max=max)
        raise argparse.ArgumentTypeError(msg)
    elif not min <= val:
        msg = "must be {min} or greater".format(min=min)
        raise argparse.ArgumentTypeError(msg)
    return val

def escape_brackets(string):
    '''使用 [[] 换替 [，使用 []] 换替 ]  (i.e. escapes [ and ] for globbing)'''
    letters = list(string)
    for i, letter in enumerate(letters[:]):
        if letter == '[':
            letters[i] = '[[]'
        elif letter == ']':
            letters[i] = '[]]'
    return ''.join(letters)

def get_input_output(arg_inputs, output_pattern="{0}.svg", ignore_duplicates=True):
    """使用 *? shell 通配符展开，得到 (input, matching output) 的遍历器

    arg_inputs: command-line-given inputs, can include *? wildcards
    output_pattern: pattern to rename output file, with {0} for input's base
        name without extension e.g. pic.png + {0}.svg = pic.svg
    ignore_duplicates: don't process or return inputs that have been returned already.
        Warning: this stores all previous inputs, so can be slow given many inputs
"""
    old_inputs = set()
    for arg_input in arg_inputs:
        if '*' in arg_input or '?' in arg_input:
            # preventing [] expansion here because glob has problems with legal [] filenames
            # ([] expansion still works in a Unix shell, it happens before Python even executes)
            if '[' in arg_input or ']' in arg_input:
                arg_input = escape_brackets(arg_input)
            inputs_ = tuple(iglob(os.path.abspath(arg_input)))
        else:
            # ensures non-existing file paths are included so they are reported as such
            # (glob silently skips over non-existing files, but we want to know about them)
            inputs_ = (arg_input,)
        for input_ in inputs_:
            if ignore_duplicates:
                if input_ not in old_inputs:
                    old_inputs.add(input_)
                    basename = os.path.basename(os.path.splitext(input_)[0])
                    output = output_pattern.format(basename)
                    yield input_, output
            else:
                basename = os.path.basename(os.path.splitext(input_)[0])
                output = output_pattern.format(basename)
                yield input_, output

def queue1_task(queue2, total, layers, settings, findex, input_file, output):
    """ 初始化文件、重新缩放、缩减颜色

    队列2: 第二个任务列表 (颜色孤立 + 临摹)
    总数: 用于测量队列二任务总数的值
    图层: 一个已经排序的列表，包含了 svg 格式的临摹图层文件
    设置: 一个字典，包含以下的键：
        colors, quantization, dither, remap, prescale, tmp
        See color_trace_multi for details of the values
    输入索引: 输入文件的整数索引 findex
    输入: 输入 png 文件
    输出: 输出 svg 路径
"""
    # 如果输出目录不存在，则创建
    output_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 临时文件会放置在各个输出文件的旁边
    scaled_file = os.path.abspath(os.path.join(settings['tmp_dir'], '{0}~scaled.png'.format(findex)))
    reduced_file = os.path.abspath(os.path.join(settings['tmp_dir'], '{0}~reduced.png'.format(findex)))
    with open(reduced_file, 'wb') as f:
        pass  # 创建一个空的减色文件

    try:
        # 如果跳过了量化，则必须使用不会增加颜色数量的缩放方法
        if settings['colors'] == 0:
            filter_ = 'point'
        else:
            filter_ = 'lanczos'
        rescale(input_file, scaled_file, settings['prescale'], filter=filter_)

        if settings['colors'] is not None: # 如果设置了颜色数量，就将原图缩减颜色
            quantize_reduce_colors(scaled_file, reduced_file, settings['colors'], algorithm=settings['quantization'], dither=settings['dither'])
        elif settings['remap'] is not None: # 如果设置了调色板图片，就将原图按调色板进行重映射
            remap_with_palette(scaled_file, reduced_file, settings['remap'], dither=settings['dither'])
        else:
            # argparse 应该已经抛出这个错误
            raise Exception("至少应该设置 'colors' 、 'remap' 中最少一个参数")
        if settings['colors'] == 1:
            color_table = ['#000000']
        else:
            color_table = make_color_table(reduced_file)

        # 基于调色板中颜色的数量更新总数
        if settings['colors'] is not None:
            total.value -= settings['colors'] - len(color_table)
        else:
            total.value -= settings['palette_colors'] - len(color_table)
        # 初始化输入索引所指文件的图层
        layers[findex] += [False] * len(color_table)

        # 得到图像宽度
        # 优先使用用户设置的宽度，如果没设置，那就去获得原来的宽度
        width = settings['width'] if settings['width'] else f'{get_width(input_file)}pt'
        height = settings['height']
        resolution = settings['resolution']

        # 添加任务到第二个任务队列
        for i, color in enumerate(color_table):
            queue2.put(
                {'width': width,
                 'height': height,
                 'resolution': resolution,
                 'color': color,
                 'palette': color_table,
                 'reduced_image': reduced_file,
                 'output_path': output,
                 'file_index': findex,
                 'color_index': i})

    except (Exception, KeyboardInterrupt) as e:
        # 发生错误时删除临时文件
        delete_files(scaled_file, reduced_file)
        raise e
    else:
        # 描摹后删除文件
        delete_files(scaled_file)

def queue2_task(layers, layer_lock, settings, width, height, resolution, color, palette, file_index, color_index, reduced_image, output_path):
    """ 分离颜色并描摹

    图层: 一个有序列表，包含了 svg 文件的临摹图层
    图层锁: 读取和写入图层对象时必须获取的锁
    设置: 一个字典，必须有以下键值:
        stack, despeckle, smoothcorners, optimizepaths, tmp
        See color_trace_multi for details of the values
    宽度: 输入图像的宽度
    颜色: 要孤立的颜色
    文件索引: 输入文件的整数索引
    颜色索引: 颜色的整数索引
    已缩减图像: 已经缩减颜色的输入图像
    输出路径: 输出路径，svg 文件
"""
    # 临时文件放在每个输出文件的旁边
    isolated_image = os.path.abspath(os.path.join(settings['tmp_dir'], '{0}-{1}~isolated.png'.format(file_index, color_index)))
    layer_file = os.path.abspath(os.path.join(settings['tmp_dir'], '{0}-{1}~layer.ppm'.format(file_index, color_index)))
    trace_format = '{0}-{1}~trace.svg'
    trace_file = os.path.abspath(os.path.join(settings['tmp_dir'], trace_format.format(file_index, color_index)))

    try:
        # 如果颜色索引是 0 并且 -bg 选项被激活
        # 直接用匹配的颜色填充图像，否则使用孤立颜色
        if color_index == 0 and settings['background']:
            Print("Index {}".format(color))
            fill_with_color(reduced_image, layer_file)
        else:
            isolate_color(reduced_image, isolated_image, layer_file, color, palette, stack=settings['stack'])
        # 描摹这个颜色，添加到 svg 栈
        trace(layer_file, trace_file, color, settings['despeckle'], settings['smoothcorners'], settings['optimizepaths'], width, height, resolution)
    except (Exception, KeyboardInterrupt) as e:
        # 若出错，则先删掉临时文件
        delete_files(reduced_image, isolated_image, layer_file, trace_file)
        raise e
    else:
        # 完成任务后删除临时文件
        delete_files(isolated_image, layer_file)

    layer_lock.acquire()
    try:
        # 添加图层
        layers[file_index][color_index] = True

        # 检查这个文件所有的图层是否都被临摹了
        is_last = False not in layers[file_index]
    finally:
        layer_lock.release()

    # 如果已经就绪，则保存 svg 文档
    if is_last:
        # 开始 svg 堆栈
        layout = svg_stack.CBoxLayout()

        traced_layers = [os.path.abspath(os.path.join(settings['tmp_dir'], trace_format.format(file_index, l))) for l in range(len(layers[file_index]))]

        # 添加图层到 svg
        for t in traced_layers:
            layout.addSVG(t)

        # 保存堆栈好的 svg 输出
        document = svg_stack.Document()
        document.setLayout(layout)
        with open(output_path, 'w') as file:
            document.save(file)

        delete_files(reduced_image, *traced_layers)


def process_task(queue1, queue2, progress, total, layers, layers_lock, settings):
    """ 处理 process 任务的函数

    q1: 第一个任务队列 (缩放 + 颜色缩减)
    q2: 第二个任务队列 (颜色隔离 + 描摹)
    progress: 第二个队列已完成任务数
    total: 第二个队列总任务数
    layers: 一个嵌套列表， layers[file_index][color_index] 是一个布尔值，
            表示 file_index 所指文件的 color_index 所指颜色的图层是否已经被描摹
    layers_lock: 读取和写入第二个任务队列中图层对象时的锁
    settings: 一个字典，必须包含下述键值:
        quantization, dither, remap, stack, prescale, despeckle, smoothcorners,
        optimizepaths, colors, tmp_dir
        See color_trace_multi for details of the values
"""
    while True:
        # 在第一个任务队列之前，从第二个人队列取一个工作，以节省临时文件和内存
        while not queue2.empty():
            try:
                task_params = queue2.get(block=False)
                queue2_task(layers, layers_lock, settings, **task_params)
                queue2.task_done()
                progress.value += 1
            except queue.Empty:
                break

        # 自第二个任务队列为空后，从第一个任务队列获取工作
        try:
            task_params = queue1.get(block=False)

            queue1_task(queue2, total, layers, settings, **task_params)
            queue1.task_done()
        except queue.Empty:
            time.sleep(.01)

        if queue2.empty() and queue1.empty():
            break


def color_trace(input_list, output_list, colors, num_processes, quantization='mc', dither=None,
         remap=None, stack=False, prescale=2, despeckle=2, smoothcorners=1.0,
         optimizepaths=0.2, background=False,
         width=None, height=None, resolution=None):
    """用指定选项彩色描摹输入图片

    输入列表: 输入文件列表，源 png 文件
    输出列表: 输出文件列表，目标 svg 文件
    颜色数: 要亮化缩减到的颜色质量，0 表示不量化
    进程数: 图像处理进程数
    quantization: 要使用的量化算法:
        - 'mc' = median-cut 中切 (默认值, 只有少量颜色, 使用 pngquant)
        - 'as' = adaptive spatial subdivision 自适应空间细分 (使用 imagemagick, 产生的颜色更少)
        - 'nq' = neuquant (生成许多颜色, 使用 pngnq)
    dither: 量化时使用的抖动拟色算法 (提醒，最后的输出结果受 despeckle 影响)
        None: 默认，不拟色
        'floydsteinberg': 当使用 'mc', 'as', 和 'nq' 时可用
        'riemersma': 只有使用 'as' 时可用
    remap：用于颜色缩减的自定义调色板图像的源（覆盖颜色数和量化）
    stack: 是否堆栈彩色描摹 (可以得到更精确的输出)
    despeckle: 抑制指定像素数量的斑点
    smoothcorners: 平滑转角: 0 表示不平滑, 1.334 为最大
        (等同于 potrace --alphamax)
    optimizepaths: 贝塞尔曲线优化: 0 最小, 5 最大
        (等同于 potrace --opttolerance)
    background：设置第一个颜色为整个 svg 背景，以减小 svg 体积
"""

    tmp_dir = tempfile.mkdtemp()

    # 新建两个任务队列
    # 第一个任务队列 = 缩放和颜色缩减
    queue1 = multiprocessing.JoinableQueue()
    # 第二个任务队列 = 颜色分离和描摹
    queue2 = multiprocessing.JoinableQueue()

    # 创建一个管理器，在两个进程时间共享图层
    manager = multiprocessing.Manager()
    layers = []
    for i in range(min(len(input_list), len(output_list))):
        layers.append(manager.list())
    # 创建一个读取和修改图层的锁
    layers_lock = multiprocessing.Lock()

    # 创建一个共享内存计数器，表示任务总数和已完成任务数
    progress = multiprocessing.Value('i', 0)
    if colors is not None:
        # 这只是一个估计值，因为量化可能会生成更少的颜色
        # 该值由第一个任务队列校正以收敛于实际总数
        total = multiprocessing.Value('i', len(layers) * colors)
    elif remap is not None:
        # 得到调色板图像的银色数量
        palette_colors = len(make_color_table(remap))
        # this is only an estimate because remapping can result in less colors
        # than in the remap variable. This value is corrected by q1 tasks to converge
        # on the real total.
        # 这只是一个估计值，因为量化可能会生成更少的颜色
        # 该值由第一个任务队列校正以收敛于实际总数
        total = multiprocessing.Value('i', len(layers) * palette_colors)
    else:
        # argparse 应当已经提前捕获这个错误
        raise Exception("应当提供 'colors' 和 'remap' 至少一个参数")

    # 创建和开始进程
    processes = []
    for i in range(num_processes):

        local = locals()
        local.pop('layers')
        local.pop('layers_lock')
        local.pop('progress')
        local.pop('total')
        local.pop('queue1')
        local.pop('queue2')
        local.pop('manager')
        local['local'] = None
        local['process'] = None
        local['processes'] = None

        process = multiprocessing.Process(target=process_task, args=(queue1, queue2, progress, total, layers, layers_lock, local))
        process.name = "color_trace worker #" + str(i)
        process.start()
        processes.append(process)

    try:
        # 对每个收入和相应的输出
        for index, (input_, output) in enumerate(zip(input_list, output_list)):
            Print(input_, ' -> ', output)

            # add a job to the first job queue
            queue1.put({'input_file': input_, 'output': output, 'findex': index})

        # show progress until all jobs have been completed
        while progress.value < total.value:
            sys.stdout.write("\r%.1f%%" % (progress.value / total.value * 100))
            sys.stdout.flush()
            time.sleep(0.25)

        sys.stdout.write("\rTracing complete!\n")

        # join the queues just in case progress is wrong
        queue1.join()
        queue2.join()
    except (Exception, KeyboardInterrupt) as e:
        # shut down subproesses
        for process in processes:
            process.terminate()
        shutil.rmtree(tmp_dir)
        raise e

    # close all processes
    for process in processes:
        process.terminate()
    shutil.rmtree(tmp_dir)


def delete_files(*filepaths):
    """如果文件存在则删除"""
    for f in filepaths:
        if os.path.exists(f):
            os.remove(f)

def get_args(cmdargs=None):
    """返回从命令行得到的参数

    cmdargs: 如果指定了，则使用这些参数，而不使用提供的脚本的参数
"""
    parser = argparse.ArgumentParser(description="使用 potrace 将位图转化为彩色 svg 矢量图",
                                     add_help=False, prefix_chars='-/')
    # 也可以通过 /? 获得帮助
    parser.add_argument(
        '-h', '--help', '/?',
        action='help',
        help="显示帮助")
    # 文件输入输出参数
    parser.add_argument('-i',
                        '--input', metavar='src', nargs='+', required=True,
                        help="输入文件，支持 * 和 ? 通配符")
    parser.add_argument('-o',
                        '--output', metavar='dest',
                        help="输出保存路径，支持 * 通配符")
    parser.add_argument('-d',
                        '--directory', metavar='destdir',
                        help="输出保存的文件夹")
    # 处理参数
    parser.add_argument('-C',
                        '--cores', metavar='N',
                        type=functools.partial(check_range, 0, None, int, "an integer"),
                        help="多进程处理的进程数 (默认使用全部核心)")
    # 尺寸参数
    parser.add_argument('--width', metavar='<dim>',
                        help="输出 svg 图像宽度，例如：6.5in、 15cm、100pt，默认单位是 inch")
    parser.add_argument('--height', metavar='<dim>',
                        help="输出 svg 图像高度，例如：6.5in、 15cm、100pt，默认单位是 inch")
    # parser.add_argument('--resolution', metavar='resolution', default='72',
    #                     help="输出 svg 图像分辨率，单位 dpi，例如：300、 300x150。默认值：72")
    # svg 文件似乎没有 dpi 概念

    # 彩色描摹选项
    # 颜色数和调色板互斥
    颜色数调色板组 = parser.add_mutually_exclusive_group(required=True)
    颜色数调色板组.add_argument('-c',
                                     '--colors', metavar='N',
                                     type=functools.partial(check_range, 0, 256, int, "an integer"),
                                     help="[若未使用 -p 参数，则必须指定该参数] "
                                          "表示在描摹前，先缩减到多少个颜色。最多 256 个。"
                                          "0表示跳过缩减颜色 (除非你的图片已经缩减过颜色，否则不推荐0)。")
    parser.add_argument('-q',
                        '--quantization', metavar='algorithm',
                        choices=('mc', 'as', 'nq'), default='mc',
                        help="颜色量化算法，即缩减颜色算法: mc, as, or nq. "
                             "'mc' (Median-Cut，中切，由 pngquant 实现，产生较少的颜色，这是默认); "
                             "'as' (Adaptive Spatial Subdivision 自适应空间细分，由 ImageMagick 实现，产生的颜色更少); "
                             "'nq' (NeuQuant 神经量化, 可以生成更多的颜色，由 pnqng 实现)。 如果 --colors 0 则不启用量化。")


    # make --floydsteinberg and --riemersma dithering mutually exclusive
    dither_group = parser.add_mutually_exclusive_group()
    dither_group.add_argument('-fs',
                              '--floydsteinberg', action='store_true',
                              help="启用 Floyd-Steinberg 拟色 (适用于所有量化算法或 -p/--palette)."
                                   "警告: 任何米色算法都会显著的增加输出 svg 图片的大小和复杂度")
    dither_group.add_argument('-ri',
                              '--riemersma', action='store_true',
                              help="启用 Rimersa 拟色 (只适用于 as 量化算法或 -p/--palette)")
    颜色数调色板组.add_argument('-r',
                                     '--remap', metavar='paletteimg',
                                     help=("使用一个自定义调色板图像，用于颜色缩减 [覆盖 -c 和 -q 选项]"))
    # image options
    parser.add_argument('-s',
                        '--stack',
                        action='store_true',
                        help="堆栈描摹 (若要更精确的输出，推荐用这个)")
    parser.add_argument('-p',
                        '--prescale', metavar='size',
                        type=functools.partial(check_range, 0, None, float, "a floating-point number"), default=1,
                        help="为得到更多的细节，在描摹前，先将图片进行缩放 (默认值: 1)。"
                             "例如使用 2，描摹前先预放大两倍")
    # potrace options
    parser.add_argument('-D',
                        '--despeckle', metavar='size',
                        type=functools.partial(check_range, 0, None, int, "an integer"), default=2,
                        help='抑制斑点的大小（单位是像素） (默认值：2)')
    parser.add_argument('-S',
                        '--smoothcorners', metavar='threshold',
                        type=functools.partial(check_range, 0, 1.334, float, "a floating-point number"), default=1.0,
                        help="转角平滑参数：0 表示不作平滑处理，1.334 是最大。（默认值：1.0")
    parser.add_argument('-O',
                        '--optimizepaths', metavar='tolerance',
                        type=functools.partial(check_range, 0, 5, float, "a floating-point number"), default=0.2,
                        help="贝塞尔曲线优化参数: 最小是0，最大是5"
                             "(默认值：0.2)")
    parser.add_argument('-bg',
                        '--background', action='store_true',
                        help=("将第一个颜色这背景色，并尽可能优化最终的 svg"))
    # other options
    parser.add_argument('-v',
                        '--verbose', action='store_true',
                        help="打印出运行时的细节")
    parser.add_argument('--version', action='version',
                        version='%(prog)s {ver}'.format(ver=version), help='显示程序版本')

    if cmdargs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmdargs)

    # with multiple inputs, --output must use at least one * wildcard
    multi_inputs = False
    for i, input_ in enumerate(get_input_output(args.input)):
        if i:
            multi_inputs = True
            break
    if multi_inputs and args.output is not None and '*' not in args.output:
        parser.error("argument -o/--output: must contain '*' wildcard when using multiple input files")

    # 'riemersma' dithering is only allowed with 'as' quantization or --palette option
    if args.riemersma:
        if args.quantization != 'as' and args.palette is None:
            parser.error("argument -ri/--riemersma: only allowed with 'as' quantization")

    return args

def main(args=None):
    """收集参数和运行描摹"""

    if args is None:
        args = get_args()

    # 设置汇报级别
    if args.verbose:
        global log_level
        log_level = 1

    # 设置输出文件名形式
    if args.output is None:
        output_format = "{0}.svg"
    elif '*' in args.output:
        output_format = args.output.replace('*', "{0}")
    else:
        output_format = args.output

    # --directory: 添加输出文件加路径
    if args.directory is not None:
        output_dir = args.directory.strip('\"\'')
        output_format = os.path.join(output_dir, output_format)

    # 如果参数没有指定的话，设置进程数
    if args.cores is None:
        try:
            num_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            Print("无法确定CPU核心数，因此假定为 1")
            num_processes = 1
    else:
        num_processes = args.cores

    # 只收集彩色描摹需要的参数
    input_output = zip(*get_input_output(args.input, output_format))
    try:
        input_list, output_list = input_output
    except ValueError:  # nothing to unpack
        input_list, output_list = [], []

    if args.floydsteinberg:
        dither = 'floydsteinberg'
    elif args.riemersma:
        dither = 'riemersma'
    else:
        dither = None

    colors = args.colors

    color_trace_args = vars(args)

    for k in ('colors', 'directory', 'input', 'output', 'cores', 'floydsteinberg', 'riemersma', 'verbose'):
        color_trace_args.pop(k)

    color_trace(input_list, output_list, colors, num_processes, dither=dither, **color_trace_args)

if __name__ == '__main__':
    main()