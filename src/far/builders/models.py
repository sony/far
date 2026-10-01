from .utils import instantiate_with_optional_ckpt
from far.models.ldwm_core import LDWMCore
from far.models.ldwm_pixel import PixelLDWM


def build_ldwm(
    tokenizer,
    generator,
    diffusion,
    memory=None,
    condition_on_context_time: bool = True,
    condition_on_context_pose: bool = True,
    retriever_loss_weight: float = 1.0,
    mask_loss_lambda: float = 0.0,
    mask_max_frac: float = 0.1,
    rag_mask_loss_lambda: float = None,
    agent_loss_weight: float = 1.0,
    agent_reward_weight: float = 1.0,
    agent_cond_binary: bool = True,
):
    tokenizer_obj = instantiate_with_optional_ckpt(tokenizer)
    in_channels, in_height, in_width = tokenizer_obj.latent_size
    generator_obj = instantiate_with_optional_ckpt(generator, in_channels=in_channels, in_height=in_height, in_width=in_width)
    diffusion_obj = instantiate_with_optional_ckpt(diffusion)
    # `num_memories` (frames per set) is no longer an __init__ arg of the memories
    # — LDWMCore supplies it per-call to `memory.forward` (1 in rag_step for a
    # ctx=2 single-document probe, context_size-1 otherwise), so don't inject it.
    memory_obj = instantiate_with_optional_ckpt(memory) if memory is not None else None
    core = LDWMCore(
        generator=generator_obj,
        diffusion=diffusion_obj,
        memory=memory_obj,
        condition_on_context_time=condition_on_context_time,
        condition_on_context_pose=condition_on_context_pose,
        retriever_loss_weight=retriever_loss_weight,
        mask_loss_lambda=mask_loss_lambda,
        mask_max_frac=mask_max_frac,
        rag_mask_loss_lambda=rag_mask_loss_lambda,
        agent_loss_weight=agent_loss_weight,
        agent_reward_weight=agent_reward_weight,
        agent_cond_binary=agent_cond_binary,
    )
    return PixelLDWM(tokenizer=tokenizer_obj, core=core)