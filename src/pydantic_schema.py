import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd
import difflib
from enum import Enum

df = pd.read_csv("data/reference/route.csv") 
active_routes = df[df['stopuse'].isnull()]['route'].unique().tolist()

try:
    if 'description' not in df.columns:
        df['description'] = ""
    
    # 篩選有效的 route
    active_df = df[df['stopuse'].isnull()].copy()
    active_routes = active_df['route'].unique().tolist()
    
    route_descriptions_text = "\n".join(
        [f"- **{row['route']}**: {row['description']}" 
         for _, row in active_df.iterrows() if pd.notna(row['description']) and row['description'].strip()]
    )

except Exception as e:
    print(f"Warning: Could not read route.csv or process descriptions: {e}")
    active_routes = []
    route_descriptions_text = ""

# 建立 Enum
RouteEnum = Enum('RouteEnum', {
    name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("/", "_"): name 
    for name in active_routes
}, type=str)

def get_route_tokens(text: str) -> set:
    """
    將 Route 字串拆解為關鍵字集合，並移除通用字詞以利比對。
    例如: "ALD+normal+EDS" -> {'ald', 'eds'}
    """
    if not text:
        return set()
    
    # 轉小寫並移除 'normal' (因為它是我們通常要替換掉的對象)
    text_clean = text.lower().replace("normal", "")
    
    # 使用常見分隔符號拆分
    tokens = set(re.split(r'[ +_\-()/]+', text_clean))
    
    # 移除空字串與目標字詞 (因為我們是用剩下的部分來找對應)
    stop_words = {''} 
    return tokens - stop_words

class SampleInfo(BaseModel):
    thought_process: str = Field(
        description=(
            "Step-by-step reasoning for this specific sample line."
            "\n1. Identify the 'Wafer ID' from the text."
            "\n2. Analyze the 'Route' vs 'Prepare' logic: "
            "Does the text mention pre-treatment (e.g., positioning, marking)? If yes, put in 'prepare'. "
            "Does it mention the main flow? If yes, match it to the RouteEnum."
            "\n3. Enum Matching: Look at the available RouteEnum options and write down exactly which one matches best BEFORE filling the route field."
        )
    )
    # raw_evidence: str = Field(
    #     description=(
    #         "The EXACT text segment or table row from the input email that describes this specific sample. "
    #         "You MUST copy it directly from the 'CURRENT TARGET CASE' content. "
    #         "Do NOT summarize, just copy. This proves you are looking at the correct input."
    #         "Keep it BRIEF (Max 30 words). Identify the specific line mentioning the Wafer ID."
    #     )
    # )

    lot_id: Optional[str] = Field(
        default=None, 
        description="SYSTEM GENERATED ID. DO NOT EXTRACT. LEAVE EMPTY."
    )
    
    wafer_id: str = Field(
        ...,
        description=(
            "Do NOT list all wafer IDs in the email; only extract the one corresponding to this specific row/item. "
            "Exclude vague terms like 'wafer' or 'die'."
            "The wafer ID or chip ID represents the code or name of the sample or die to be tested. "
            "There may be multiple wafer IDs in a single email for this specific sample entry, and there's no fixed format. "
            "Possible names include: 2538ESNE013.000, OPL17-003, MSS_ES_Kangyi_TT2-SN-014, ALD+1:1:10 test-2, OD-6-S1-Base, etc."
            "\n[Contextual Disambiguation]: "
            "Look for patterns typical of Lot IDs, serial numbers, or experimental labels (often alphanumeric sequences with dashes or dots)."
        )
    )
    
    # route: Optional[RouteEnum] = Field(
    #     default=None,
    #     description=(
    #         "The standard process route code. "
    #         "Refers to the specific sequence and combination of manufacturing processes that a sample must undergo to satisfy the customer's technical requirements and specifications."
    #         "You MUST select the closest matching value from the provided Enum list. "
    #         "If the email text is slightly different (e.g., 'Probing T006' vs 'probing- T006'), "
    #         "automatically map it to the correct Enum value."
    #         "\n[Route Knowledge Base]: Use the definitions below to help identify the correct route:"
    #         f"\n{route_descriptions_text}"
    #         "\n[Differentiation Rule]: "
    #         "Strictly distinguish 'Route' from 'Prepare'. "
    #         "- 'Route' is the main manufacturing/analysis flow (e.g., Standard TEM flow, WAT recipe). "
    #         "- 'Prepare' involves pre-treatment steps (e.g., E-Pt, marking, DB-positioning). "
    #         "Do NOT classify pre-treatment steps as the Route unless they are explicitly part of the standard process flow definition defined in the Enum."
    #     )
    # )
    route: Optional[str] = Field(
        default=None,
        description=(
            "The standard process route code. "
            "Refers to the specific sequence and combination of manufacturing processes that a sample must undergo."
            "CRITICAL: You MUST output exactly one of the valid route names from the Route Knowledge Base below:\n"
            f"\n{route_descriptions_text}"
        ),
        json_schema_extra={
            "enum": active_routes
        }
    )

    # @field_validator('route', mode='before')
    # @classmethod
    # def fuzzy_match_route(cls, v):
    #     if not v:
    #         return v
        
    #     input_str = str(v).strip()
    #     valid_options = active_routes

    #     if input_str in valid_options:
    #         return input_str

    #     for option in valid_options:
    #         if input_str.lower() == option.lower():
    #             return option
                
    #     # Lower cutoff to 0.6 to be more forgiving, and return the raw string if it fails instead of crashing
    #     matches = difflib.get_close_matches(input_str, valid_options, n=1, cutoff=0.6)
    #     if matches:
    #         return matches[0]

    #     return input_str
    
    prepare: Optional[str] = Field(
        default=None,
        description="""
            [Industry Context: Semiconductor FA/Process]
            Defines pre-treatment steps BEFORE the main analysis route. **Find the "Receipe"**.

            [MANDATORY MAPPING RULES]:
            1. **Priority 1 (Explicit Text):** Extract exact parameters if present (e.g., 'ALD(W2-A)', 'M-bond(60/30)').
               **MUST NOT** write "Coating", "epoxy", "Resin", or "Glue" only.
               **MUST NOT** write "Coating", "epoxy", "Resin", or "Glue" only.
            2. **Priority 2 (History Inference - CRITICAL):** 
               **MUST** look at the 'HISTORICAL REFERENCE CASES' to find the specific recipe used for this customer/macro (e.g., 'ALD(W2 35cycle)', 'Pi bond(60/30)', or 'M-bond(60/30)') and apply it to the current sample.

            [Output Format]
            Combine steps with '+': "Top view+ALD(W2-A)+DB+Pi bond(60/30)+Probing"
        """
    )
    
    loctestkey: Optional[str] = Field(
        default=None,
        description=(
            "Specifies the exact target location/coordinates or Macro Name on the sample. "
            
            "Key Indicators & Patterns to Extract: "
            "1. **Directional Cuts:** explicit mentions of 'X-CUT' (X-direction cross-section) or 'Y-CUT' (Y-direction cross-section). "
            "   - Examples to generate: '23P Y-CUT', 'Pt 5s-XCUT', 'Macro 3G Y-cut'. "
            "2. **Coordinates/Position Codes:** numerical sequences indicating specific die or structure locations. "
            "   - Examples to generate: '60-20-03', 'R3-C5'. "
            "3. **Feature Targeting:** references to locations identified by inspection tools. "
            "   - Examples to generate: 'AOI' (target the defect found by AOI), 'Hotspot'. "
            "4. **Macro Name: ** If a 'Macro', 'Navi Map', or 'Target' column exists (e.g., 'XCH', 'Macro 3G Y-cut', 'Spark Ycut 40P', 'DUT2 X-Cut'), use this EXACT string as the loctestkey. "

            "[Constraints]: Do NOT confuse with 'wafer_id'. Remove filenames like .pdf or .pptx."
        )
    )

class Stage1Sample(BaseModel):
    wafer_id: str = Field(
        ...,
        description=(
            "Do NOT list all wafer IDs in the email; only extract the one corresponding to this specific row/item. "
            "Exclude vague terms like 'wafer' or 'die'. "
            "The wafer ID or chip ID represents the code or name of the sample or die to be tested."
        )
    )

class Stage1Order(BaseModel):
    """階段一專用：僅抽取全局資訊與實體切分"""
    global_analysis: str = Field(
        description="Explicitly state exactly how many Wafer IDs/Samples are present in the text."
    )
    company: Optional[str] = Field(
        default=None,
        description="Identifies the customer's company or organization name. DO NOT EXTRACT MSS. Same rules as before."
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="The specific contact person's name representing the customer company."
    )
    samples: List[Stage1Sample] = Field(
        default_factory=list,
        description="A list of unique samples. CRITICAL: DO NOT output duplicate Wafer IDs. Stop generating when all unique IDs are listed."
    )

    # global_analysis: str = Field(
    #     description=(
    #         "Execute a Deep Logic Analysis step before extraction:"
    #         "\n1. **Count Samples:** Explicitly scan the text and tables in the <target_case> and state EXACTLY how many Wafer IDs/Samples are present."
    #         "\n2. **List all Samples:** Identify each unique sample or Wafer ID present in the 'CURRENT TARGET CASE'."
    #         "\n3. **Map Specific Instructions (CRITICAL):** For EACH sample, explicitly state what the text or PPT requested. "
    #         "(e.g., 'Sample A requires ALD+DB+Probing because of X-Cut note. Sample B only requires ALD+Topview')."
    #         "\n4. **Do Not Generalize:** Pay close attention to row-specific or item-specific notes. "
    #         "DO NOT assume all samples share the same prep or route conditions unless explicitly stated as 'applies to all'."
    #         "\n5. **Infer Hidden Steps (Domain Knowledge):**"
    #         "   - 'Easy-Lift' / 'Nano-probing' implies Route='Probing'."
    #         "   - 'Epoxy' or 'Glue' implies Prepare='M-bond'."
    #     )
    # )
    
    # company: Optional[str] = Field(
    #     default=None,
    #     description=(
    #         "Identifies the customer's company or organization name associated with this request. "
    #         "\n[ROLE CONTEXT]: You represent 'MSS' (MSSCORPS / Panquan Technology), the testing VENDOR. The 'customer' is the external client sending you samples."
    #         "\n**CRITICAL NEGATIVE RULE**: NEVER output 'MSS', 'MSSCORPS', 'MSS USA', or 'Panquan Technology'. Look for the external company (e.g., 'TEL', 'TSMC', 'Nvidia')."
    #         "\nExtraction Logic: "
    #         "1. **Email Domain Parsing:** 'user@nvidia.com' -> 'NVIDIA'."
    #         "2. **TSMC Rule:** If 'TSMC' or 'Taiwan Semiconductor', output '230'."
    #     )
    # )
    
    # customer_name: Optional[str] = Field(
    #     default=None,
    #     description=(
    #         "The specific contact person's name representing the customer company. "
    #         "\n[CRITICAL NEGATIVE RULE]: NEVER extract internal MSS employees. Known MSS employees to IGNORE: 'Katie', 'Chen Jiaxin', 'Amy', 'David Lo', 'Jimmy', 'Nurul', 'PP', 'TEM'."
    #         "\nExtract the sender from the external customer company (e.g., 'Kang-Yi Lin', 'Hailey Jenkins')."
    #     )
    # )
    
    # samples: List[SampleInfo] = Field(
    #     default_factory=list,
    #     description=(
    #         "A comprehensive list of all individual samples or test items identified in the email. "
    #         "This is the core collection of the order."
    #         "\n\nParsing Strategy:"
    #         "\n1. **Table/List Iteration:** Treat each row in a data table or each bullet point in the email body as a potential independent `SampleInfo` object."
    #         "\n2. **One Object per Entity:** Even if multiple Wafer IDs are listed in a single line, if they represent distinct physical samples, split them into separate objects."
    #         # "\n3. **Global vs. Local Context (Inheritance):** "
    #         # "   - If instructions (e.g., 'Route', 'Prepare') are stated at the top of the email or table header as 'applies to all', "
    #         # "     propagate this information to every `SampleInfo` object unless a specific row overrides it."
    #     )
    # )

    @field_validator('company')
    @classmethod
    def normalize_company_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None: return v
        clean_v = v.strip().upper()
        tsmc_variants = ["TSMC", "TAIWAN SEMICONDUCTOR", "T.S.M.C", "台積電"]
        if any(variant in clean_v for variant in tsmc_variants):
            return "230"
        return v

class Stage2Inference(BaseModel):
    """階段二專用：針對單一樣本進行邏輯推論"""
    # thought_process: str = Field(
    #     description=(
    #         "Execute a Deep Logic Analysis step before extraction:"
    #         "\n1. **Map Specific Instructions (CRITICAL):** For EACH sample, explicitly state what the text or PPT requested. "
    #         "\n2. **Do Not Generalize:** Pay close attention to row-specific or item-specific notes. "
    #         "DO NOT assume all samples share the same prep or route conditions unless explicitly stated as 'applies to all'."
    #         "\n3. **Infer Hidden Steps (Domain Knowledge):**"
    #         "   - 'Easy-Lift' / 'Nano-probing' implies Route='Probing'."
    #         "   - 'Epoxy' or 'Glue' implies Prepare='M-bond'."
    #     )
    # )
    route: Optional[str] = Field(
        default=None,
        description=(
            "The standard process route code. "
            "Refers to the specific sequence and combination of manufacturing processes that a sample must undergo."
            "CRITICAL: You MUST output exactly one of the valid route names from the Route Knowledge Base below:\n"
            f"\n{route_descriptions_text}"
        ),
        json_schema_extra={
            "enum": active_routes
        }
    )

    prepare: Optional[str] = Field(
        default=None,
        description="""
            [Industry Context: Semiconductor FA/Process]
            Defines pre-treatment steps BEFORE the main analysis route. **Find the "Receipe"**.

            [MANDATORY MAPPING RULES]:
            1. **Priority 1 (Explicit Text):** Extract exact parameters if present (e.g., 'ALD(W2-A)', 'M-bond(60/30)').
               **MUST NOT** write "Coating", "epoxy", "Resin", or "Glue" on any step
               **MUST NOT** write "Coating", "epoxy", "Resin", or "Glue" on any step
               For example, **MUST NOT** write "ALD(W2-A)+epoxy".
            2. **Priority 2 (History Inference - CRITICAL):** 
               **MUST** look at the 'HISTORICAL REFERENCE CASES' to find the specific recipe used for this customer/macro (e.g., 'ALD(W2 35cycle)', 'Pi bond(60/30)', or 'M-bond(60/30)') and apply it to the current sample.

            [Output Format]
            Combine steps with '+': "Top view+ALD(W2-A)+DB+Pi bond(60/30)+Probing"
        """
    )
    loctestkey: Optional[str] = Field(
        default=None,
        description=(
            "Specifies the exact target location/coordinates or Macro Name on the sample. "
            
            "Key Indicators & Patterns to Extract: "
            "1. **Directional Cuts:** explicit mentions of 'X-CUT' (X-direction cross-section) or 'Y-CUT' (Y-direction cross-section). "
            "   - Examples to generate: '23P Y-CUT', 'Pt 5s-XCUT', 'Macro 3G Y-cut'. "
            "2. **Coordinates/Position Codes:** numerical sequences indicating specific die or structure locations. "
            "   - Examples to generate: '60-20-03', 'R3-C5'. "
            "3. **Feature Targeting:** references to locations identified by inspection tools. "
            "   - Examples to generate: 'AOI' (target the defect found by AOI), 'Hotspot'. "
            "4. **Macro Name: ** If a 'Macro', 'Navi Map', or 'Target' column exists (e.g., 'XCH', 'Macro 3G Y-cut', 'Spark Ycut 40P', 'DUT2 X-Cut'), use this EXACT string as the loctestkey. "

            "[Constraints]: Do NOT confuse with 'wafer_id'. Remove filenames like .pdf or .pptx."
        )
    )

    # 沿用原有的驗證邏輯
    @field_validator('prepare')
    @classmethod
    def clean_prepare_logic(cls, v: Optional[str]) -> Optional[str]:
        if not v: return v
        raw_steps = [step.strip() for step in re.split(r'[+,]', v)]
        cleaned_steps = []
        for step in raw_steps:
            step = step.strip()
            if not step: continue
            if 'top' in step.lower() and 'view' in step.lower():
                step = "Top view"
            cleaned_steps.append(step)
        unique_steps = []
        seen = set()
        for s in cleaned_steps:
            s_key = s.lower().replace(" ", "")
            if s_key not in seen:
                unique_steps.append(s)
                seen.add(s_key)
        return "+".join(unique_steps)

    @field_validator('loctestkey')
    @classmethod
    def clean_loctestkey(cls, v: str) -> str:
        if not v: return v
        v = re.sub(r'\.(pdf|pptx?|docx?|txt|xlsx?|jpg|png)$', '', v, flags=re.IGNORECASE)
        v = re.sub(r'\s*\(.*?\)', '', v)
        v = v.strip(' |-,')
        return v.strip()
    

class OrderInfo(BaseModel):
    """
    代表從電子郵件中提取的整個訂單資訊。
    包含郵件頭部的元數據 (Metadata) 以及樣品列表。
    """
    global_analysis: str = Field(
        description=(
            "Execute a Deep Logic Analysis step before extraction:"
            "\n1. **Count Samples:** Explicitly scan the text and tables in the <target_case> and state EXACTLY how many Wafer IDs/Samples are present."
            "\n2. **List all Samples:** Identify each unique sample or Wafer ID present in the 'CURRENT TARGET CASE'."
            "\n3. **Map Specific Instructions (CRITICAL):** For EACH sample, explicitly state what the text or PPT requested. "
            "(e.g., 'Sample A requires ALD+DB+Probing because of X-Cut note. Sample B only requires ALD+Topview')."
            "\n4. **Do Not Generalize:** Pay close attention to row-specific or item-specific notes. "
            "DO NOT assume all samples share the same prep or route conditions unless explicitly stated as 'applies to all'."
            "\n5. **Infer Hidden Steps (Domain Knowledge):**"
            "   - 'Easy-Lift' / 'Nano-probing' implies Route='Probing'."
            "   - 'Epoxy' or 'Glue' implies Prepare='M-bond'."
        )
    )
    
    company: Optional[str] = Field(
        default=None,
        description=(
            "Identifies the customer's company or organization name associated with this request. "
            "\n[ROLE CONTEXT]: You represent 'MSS' (MSSCORPS / Panquan Technology), the testing VENDOR. The 'customer' is the external client sending you samples."
            "\n**CRITICAL NEGATIVE RULE**: NEVER output 'MSS', 'MSSCORPS', 'MSS USA', or 'Panquan Technology'. Look for the external company (e.g., 'TEL', 'TSMC', 'Nvidia')."
            "\nExtraction Logic: "
            "1. **Email Domain Parsing:** 'user@nvidia.com' -> 'NVIDIA'."
            "2. **TSMC Rule:** If 'TSMC' or 'Taiwan Semiconductor', output '230'."
        )
    )
    
    customer_name: Optional[str] = Field(
        default=None,
        description=(
            "The specific contact person's name representing the customer company. "
            "\n[CRITICAL NEGATIVE RULE]: NEVER extract internal MSS employees. Known MSS employees to IGNORE: 'Katie', 'Chen Jiaxin', 'Amy', 'David Lo', 'Jimmy', 'Nurul', 'PP', 'TEM'."
            "\nExtract the sender from the external customer company (e.g., 'Kang-Yi Lin', 'Hailey Jenkins')."
        )
    )
    
    samples: List[SampleInfo] = Field(
        default_factory=list,
        description=(
            "A comprehensive list of all individual samples or test items identified in the email. "
            "This is the core collection of the order."
            "\n\nParsing Strategy:"
            "\n1. **Table/List Iteration:** Treat each row in a data table or each bullet point in the email body as a potential independent `SampleInfo` object."
            "\n2. **One Object per Entity:** Even if multiple Wafer IDs are listed in a single line, if they represent distinct physical samples, split them into separate objects."
            # "\n3. **Global vs. Local Context (Inheritance):** "
            # "   - If instructions (e.g., 'Route', 'Prepare') are stated at the top of the email or table header as 'applies to all', "
            # "     propagate this information to every `SampleInfo` object unless a specific row overrides it."
        )
    )

    # @field_validator('company')
    # @classmethod
    # def normalize_company_code(cls, v: Optional[str]) -> Optional[str]:
    #     if v is None:
    #         return v
        
    #     # 移除前後空白並轉為大寫進行比對
    #     clean_v = v.strip().upper()
        
    #     # 定義 TSMC 的常見變體
    #     tsmc_variants = ["TSMC", "TAIWAN SEMICONDUCTOR", "T.S.M.C", "台積電"]
        
    #     # 檢查是否包含關鍵字 (模糊比對) 或是完全匹配
    #     # 這裡採取嚴格策略：只要字串中包含 TSMC，就強制轉為 230
    #     if any(variant in clean_v for variant in tsmc_variants):
    #         return "230"
            
    #     return v

# # --- 測試區塊 (只在直接執行此檔案時運作) ---
# if __name__ == "__main__":
#     try:
#         # 簡單打印出 JSON Schema 來驗證結構是否正確
#         print("Schema 驗證成功！生成的 JSON Schema 如下：")
#         print(OrderInfo.model_json_schema())
#     except Exception as e:
#         print(f"Schema 定義有誤: {e}")