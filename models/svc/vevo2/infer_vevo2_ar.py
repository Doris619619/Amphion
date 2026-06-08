import os
import torch
from huggingface_hub import snapshot_download

from models.svc.vevo2.vevo2_utils import *


def vevo2_tts(
    tgt_text,
    ref_wav_path,
    ref_text=None,
    timbre_ref_wav_path=None,
    output_path=None,
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        target_text=tgt_text,
        style_ref_wav_path=ref_wav_path,
        style_ref_wav_text=ref_text,
        timbre_ref_wav_path=timbre_ref_wav_path,
        use_prosody_code=False,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def vevo2_editing(
    tgt_text,
    raw_wav_path,
    raw_text=None,
    output_path=None,
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        target_text=tgt_text,
        prosody_wav_path=raw_wav_path,
        style_ref_wav_path=raw_wav_path,
        style_ref_wav_text=raw_text,
        timbre_ref_wav_path=raw_wav_path,
        use_prosody_code=True,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def vevo2_singing_style_conversion(
    raw_wav_path,
    style_ref_wav_path,
    output_path=None,
    raw_text=None,
    style_ref_text=None,
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        target_text=raw_text,
        prosody_wav_path=raw_wav_path,
        style_ref_wav_path=style_ref_wav_path,
        style_ref_wav_text=style_ref_text,
        timbre_ref_wav_path=raw_wav_path,
        use_prosody_code=True,
        use_pitch_shift=True,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def vevo2_melody_control(
    tgt_text,
    tgt_melody_wav_path,
    output_path=None,
    style_ref_wav_path=None,
    style_ref_text=None,
    timbre_ref_wav_path=None,
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        target_text=tgt_text,
        prosody_wav_path=tgt_melody_wav_path,
        style_ref_wav_path=style_ref_wav_path,
        style_ref_wav_text=style_ref_text,
        timbre_ref_wav_path=timbre_ref_wav_path,
        use_prosody_code=True,
        use_pitch_shift=True,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)


def load_inference_pipeline():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    local_dir = snapshot_download(
        repo_id="RMSnow/Vevo2",
        repo_type="model",
        local_dir="./ckpts/Vevo2",
        resume_download=True,
    )

    content_style_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/contentstyle_fvq16384_12.5hz"
    )
    prosody_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/prosody_fvq512_6.25hz"
    )

    ar_cfg_path = os.path.join(
        local_dir, "contentstyle_modeling/posttrained/amphion_config.json"
    )
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/posttrained")

    fmt_cfg_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json"
    )
    fmt_ckpt_path = os.path.join(
        local_dir, "acoustic_modeling/fm_emilia101k_singnet7k_repa"
    )

    vocoder_cfg_path = os.path.join(local_dir, "vocoder/config.json")
    vocoder_ckpt_path = os.path.join(local_dir, "vocoder")

    inference_pipeline = Vevo2InferencePipeline(
        prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    return inference_pipeline


if __name__ == "__main__":
    inference_pipeline = load_inference_pipeline()

    output_dir = "./models/svc/vevo2/output"
    os.makedirs(output_dir, exist_ok=True)

    ### Zero-shot Text-to-Speech and Text-to-Singing  ###
    tgt_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."
    #ref_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    #这里控制音色
    ref_wav_path = "/root/autodl-tmp/cuhk/en_sample_0_18s_24k.wav"
    ref_text = "Flip stood undecided, his ears strained to catch the slightest sound."

    jaychou_path = "./models/svc/vevosing/wav/jaychou.wav"
    jaychou_text = (
        "对这个世界如果你有太多的抱怨，跌倒了就不该继续往前走，为什么，人要这么的脆弱堕"
    )
    taiyizhenren_path = "./models/svc/vevosing/wav/taiyizhenren.wav"
    taiyizhenren_text = (
        "对，这就是我，万人敬仰的太乙真人。虽然有点婴儿肥，但也掩不住我，逼人的帅气。"
    )

    # the style reference and timbre reference are same
    vevo2_tts(
        #tgt_text=tgt_text,
        #ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=ref_wav_path,
        output_path=os.path.join(output_dir, "zstts.wav"),
        #ref_text=ref_text,
        tgt_text = "我们来看第一小题。这里的曲线是 y 等于一除以 x，题目要求我们求它在 x 等于 a 这一点的斜率，其中 a 不能等于零。按照导数的定义，我们要先比较 f 的 a 加 h 和 f 的 a，再除以 h，然后让 h 趋近于零。化简之后，分子会变成负 h，分母是 h 乘以 a，再乘以 a 加 h。这里的 h 可以约掉，所以最后的极限就是负的一除以 a 的平方。因此，在 x 等于 a 的地方，斜率是负的一除以 a 平方。特别地，当 x 等于负一时，斜率就是负一。",
        #tgt_text = "Let us look at part a. The curve is y equals one over x, and we want to find its slope at a general point x equals a, where a is not zero. Using the definition of the derivative, we compare f of a plus h with f of a, divide by h, and then let h approach zero. After simplifying, the difference becomes negative h over h times a times a plus h. The h cancels, and the limit is negative one over a squared. So the slope at x equals a is negative one over a squared. In particular, when x equals negative one, the slope is negative one.",
        
        
        #ref_wav_path = "/root/autodl-tmp/cuhk/puremath_24k.wav",

        #ref_text = "而纯数学基本是猜想来决定的，就是数学里面很多猜想。做纯数学的，包括我和我的同学，做纯数学的，他们都是解决猜想。啊这个，这是两种不同的这个这个这个这个 motivation。",
        

        #WCM
        #ref_wav_path = "/root/autodl-tmp/cuhk/en_sample_0_18s_24k.wav",

        #ref_text = "Good afternoon, everyone. It is my pleasure to give this talk at MWC 2025. This talk is about a line of work that I have studied with my colleagues at the Shenzhen Research Institute of Big Data and The Chinese University of Hong Kong, Shenzhen.",
    
        ref_wav_path = "/root/autodl-tmp/cuhk/puremath_24k.wav",

        ref_text = "而纯数学基本是猜想来决定的，就是数学里面很多猜想。做纯数学的，包括我和我的同学，做纯数学的，他们都是解决猜想。啊这个，这是两种不同的这个这个这个这个 motivation。",
    
    
    
    )

    # the style reference and timbre reference are different
    vevo2_tts(
        tgt_text=tgt_text,
        ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=jaychou_path,
        output_path=os.path.join(output_dir, "zstts_disentangled.wav"),
        ref_text=ref_text,
    )

    # the style reference is a singing voice
    vevo2_tts(
        tgt_text="顿时，气氛变得沉郁起来。乍看之下，一切的困扰仿佛都围绕在我身边。我皱着眉头，感受着那份压力，但我知道我不能放弃，不能认输。于是，我深吸一口气，心底的声音告诉我：“无论如何，都要冷静下来，重新开始。”",
        ref_wav_path=jaychou_path,
        ref_text=jaychou_text,
        timbre_ref_wav_path=taiyizhenren_path,
        output_path=os.path.join(output_dir, "zstts_singing.wav"),
    )

    ### Zero-shot Singing Editing ###
    adele_path = "./models/svc/vevosing/wav/adele.wav"
    adele_text = "Never mind, I'll find someone like you. I wish nothing but."

    vevo2_editing(
        tgt_text="Never mind, you'll find anyone like me. You wish nothing but.",
        raw_wav_path=adele_path,
        raw_text=adele_text,  # "Never mind, I'll find someone like you. I wish nothing but."
        output_path=os.path.join(output_dir, "editing_adele.wav"),
    )

    vevo2_editing(
        tgt_text="对你的人生如果你有太多的期盼，跌倒了就不该低头认输，为什么啊，人要这么的彷徨堕",
        raw_wav_path=jaychou_path,
        raw_text=jaychou_text,  # "对这个世界如果你有太多的抱怨，跌倒了就不该继续往前走，为什么，人要这么的脆弱堕"
        output_path=os.path.join(output_dir, "editing_jaychou.wav"),
    )

    ### Zero-shot Singing Style Conversion ###
    breathy_path = "./models/svc/vevosing/wav/breathy.wav"
    breathy_text = "离别没说再见你是否心酸"

    vibrato_path = "./models/svc/vevosing/wav/vibrato.wav"
    vibrato_text = "玫瑰的红，容易受伤的梦，握在手中却流失于指缝"

    vevo2_singing_style_conversion(
        raw_wav_path=breathy_path,
        raw_text=breathy_text,
        style_ref_wav_path=vibrato_path,
        style_ref_text=vibrato_text,
        output_path=os.path.join(output_dir, "ssc_breathy2vibrato.wav"),
    )

    ### Melody Control for Singing Synthesis ##
    humming_path = "./models/svc/vevosing/wav/humming.wav"
    piano_path = "./models/svc/vevosing/wav/piano.wav"

    # Humming to control the melody
    vevo2_melody_control(
        tgt_text="你是我的小呀小苹果，怎么爱，不嫌多",
        tgt_melody_wav_path=humming_path,
        output_path=os.path.join(output_dir, "melody_humming.wav"),
        style_ref_wav_path=taiyizhenren_path,
        style_ref_text=taiyizhenren_text,
        timbre_ref_wav_path=taiyizhenren_path,
    )

    # Piano to control the melody
    vevo2_melody_control(
        tgt_text="你是我的小呀小苹果，怎么爱，不嫌多",
        tgt_melody_wav_path=piano_path,
        output_path=os.path.join(output_dir, "melody_piano.wav"),
        style_ref_wav_path=taiyizhenren_path,
        style_ref_text=taiyizhenren_text,
        timbre_ref_wav_path=taiyizhenren_path,
    )
