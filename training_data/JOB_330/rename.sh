#!/bin/bash

# 定义所有目标文件夹名
folders=(
job_1 job_10 job_100 job_101 job_102 job_103 job_104 job_105 job_106 job_107 job_108 job_109
job_11 job_110 job_111 job_112 job_113 job_114 job_115 job_116 job_117 job_118 job_119 job_12
job_120 job_121 job_122 job_123 job_124 job_125 job_126 job_127 job_128 job_129 job_13 job_130
job_131 job_132 job_133 job_134 job_135 job_136 job_137 job_138 job_139 job_14 job_140 job_141
job_142 job_143 job_144 job_145 job_146 job_147 job_148 job_149 job_15 job_150 job_151 job_152
job_153 job_154 job_155 job_156 job_157 job_158 job_159 job_16 job_160 job_161 job_162 job_163
job_164 job_165 job_166 job_167 job_168 job_169 job_17 job_170 job_171 job_172 job_173 job_174
job_175 job_176 job_177 job_178 job_179 job_18 job_180 job_181 job_182 job_183 job_184 job_185
job_186 job_187 job_188 job_189 job_19 job_190 job_191 job_192 job_193 job_194 job_195 job_196
job_197 job_198 job_199 job_2 job_20 job_200 job_201 job_202 job_203 job_204 job_205 job_206
job_207 job_208 job_209 job_21 job_210 job_211 job_212 job_213 job_214 job_215 job_216 job_217
job_218 job_219 job_22 job_220 job_221 job_222 job_223 job_224 job_225 job_226 job_227 job_228
job_229 job_23 job_230 job_231 job_232 job_233 job_234 job_235 job_236 job_237 job_238 job_239
job_24 job_240 job_241 job_242 job_243 job_244 job_245 job_246 job_247 job_248 job_249 job_25
job_250 job_251 job_252 job_253 job_254 job_255 job_256 job_257 job_258 job_259 job_26 job_260
job_261 job_262 job_263 job_264 job_265 job_266 job_267 job_268 job_269 job_27 job_270 job_271
job_272 job_273 job_274 job_275 job_276 job_277 job_278 job_279 job_28 job_280 job_281 job_282
job_283 job_284 job_285 job_286 job_287 job_288 job_289 job_29 job_290 job_291 job_292 job_293
job_294 job_295 job_296 job_297 job_298 job_299 job_3 job_30 job_300 job_301 job_302 job_303
job_304 job_305 job_306 job_307 job_308 job_309 job_31 job_310 job_311 job_312 job_313 job_314
job_315 job_316 job_317 job_318 job_319 job_32 job_320 job_321 job_322 job_323 job_324 job_325
job_326 job_327 job_328 job_329 job_33 job_330 job_34 job_35 job_36 job_37 job_38 job_39 job_4
job_40 job_41 job_42 job_43 job_44 job_45 job_46 job_47 job_48 job_49 job_5 job_50 job_51 job_52
job_53 job_54 job_55 job_56 job_57 job_58 job_59 job_6 job_60 job_61 job_62 job_63 job_64 job_65
job_66 job_67 job_68 job_69 job_7 job_70 job_71 job_72 job_73 job_74 job_75 job_76 job_77 job_78
job_79 job_8 job_80 job_81 job_82 job_83 job_84 job_85 job_86 job_87 job_88 job_89 job_9 job_90
job_91 job_92 job_93 job_94 job_95 job_96 job_97 job_98 job_99
)

# 批量重命名
for folder in "${folders[@]}"; do
    if [ -f "$folder/cost_matrix_10.json" ]; then
        mv "$folder/cost_matrix_10.json" "$folder/cost_matrix_30.json"
    fi
done

echo "重命名完成。"

