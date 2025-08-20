set(FLASH_ATTENTION_URL https://github.com/Oneflow-Inc/flash-attention-v2/archive/eed2e82b880e06237af3e50ceac4cf6728b15645.zip)
use_mirror(VARIABLE FLASH_ATTENTION_URL URL ${FLASH_ATTENTION_URL})
set(FLASH_ATTENTION_MD5 cff17aa2a95ad754c9889bb3a03d5c51)

CPMAddPackage(
    NAME flash_attention
    URL ${FLASH_ATTENTION_URL}
    URL_HASH MD5=${FLASH_ATTENTION_MD5}
)