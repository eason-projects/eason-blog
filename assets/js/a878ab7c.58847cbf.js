"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([[2026],{3881:(n,e,i)=>{i.d(e,{R:()=>r,x:()=>t});var s=i(8101);const l={},d=s.createContext(l);function r(n){const e=s.useContext(d);return s.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function t(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(l):n.components||l:r(n.components),s.createElement(d.Provider,{value:e},n.children)}},5550:n=>{n.exports=JSON.parse('{"permalink":"/eason-blog/blog/ble-beacon","editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/blog/2025-03-09-ble-beacon/index.md","source":"@site/blog/2025-03-09-ble-beacon/index.md","title":"\u4f7f\u7528Python\u68c0\u6d4b\u84dd\u7259\u4fe1\u53f7","description":"\u672c\u6587\u4ecb\u7ecd\u4e86\u5982\u4f55\u4f7f\u7528Python\u5728MacOS\u4e0a\u68c0\u6d4bBLE\u4fe1\u53f7\u5e76\u53ef\u89c6\u5316\u5c55\u793a\u4fe1\u53f7\u5f3a\u5ea6\u3002","date":"2025-03-09T00:00:00.000Z","tags":[{"inline":false,"label":"Bluetooth","permalink":"/eason-blog/blog/tags/bluetooth","description":"Bluetooth related posts"}],"readingTime":13.85,"hasTruncateMarker":true,"authors":[{"name":"Eason G.","title":"Engineer","url":"https://github.com/e10101","page":{"permalink":"/eason-blog/blog/authors/eason"},"socials":{"github":"https://github.com/e10101"},"imageURL":"https://github.com/e10101.png","key":"eason"}],"frontMatter":{"slug":"ble-beacon","title":"\u4f7f\u7528Python\u68c0\u6d4b\u84dd\u7259\u4fe1\u53f7","authors":"eason","tags":["bluetooth"],"draft":false},"unlisted":false,"nextItem":{"title":"\u6c49\u5b57\u7b14\u987a\u9884\u6d4b\u6a21\u578b","permalink":"/eason-blog/blog/2025/02/24/stroke-order/README"}}')},6594:(n,e,i)=>{i.r(e),i.d(e,{assets:()=>a,contentTitle:()=>t,default:()=>h,frontMatter:()=>r,metadata:()=>s,toc:()=>c});var s=i(5550),l=i(5105),d=i(3881);const r={slug:"ble-beacon",title:"\u4f7f\u7528Python\u68c0\u6d4b\u84dd\u7259\u4fe1\u53f7",authors:"eason",tags:["bluetooth"],draft:!1},t=void 0,a={authorsImageUrls:[void 0]},c=[{value:"\u5e38\u89c1\u7684\u5b9a\u4f4d\u65b9\u6cd5",id:"\u5e38\u89c1\u7684\u5b9a\u4f4d\u65b9\u6cd5",level:2},{value:"UWB\uff08Ultra-Wideband\uff0c\u8d85\u5bbd\u5e26\uff09\u5b9a\u4f4d",id:"uwbultra-wideband\u8d85\u5bbd\u5e26\u5b9a\u4f4d",level:3},{value:"\u5de5\u4f5c\u539f\u7406",id:"\u5de5\u4f5c\u539f\u7406",level:4},{value:"\u4f18\u52bf",id:"\u4f18\u52bf",level:4},{value:"\u52a3\u52bf",id:"\u52a3\u52bf",level:4},{value:"\u5e94\u7528\u573a\u666f",id:"\u5e94\u7528\u573a\u666f",level:4},{value:"BLE\uff08Bluetooth Low Energy\uff0c\u4f4e\u529f\u8017\u84dd\u7259\uff09\u5b9a\u4f4d",id:"blebluetooth-low-energy\u4f4e\u529f\u8017\u84dd\u7259\u5b9a\u4f4d",level:3},{value:"\u5de5\u4f5c\u539f\u7406",id:"\u5de5\u4f5c\u539f\u7406-1",level:4},{value:"\u4f18\u52bf",id:"\u4f18\u52bf-1",level:4},{value:"\u52a3\u52bf",id:"\u52a3\u52bf-1",level:4},{value:"\u5e94\u7528\u573a\u666f",id:"\u5e94\u7528\u573a\u666f-1",level:4},{value:"WiFi\u5b9a\u4f4d",id:"wifi\u5b9a\u4f4d",level:3},{value:"\u5de5\u4f5c\u539f\u7406",id:"\u5de5\u4f5c\u539f\u7406-2",level:4},{value:"\u4f18\u52bf",id:"\u4f18\u52bf-2",level:4},{value:"\u52a3\u52bf",id:"\u52a3\u52bf-2",level:4},{value:"\u5e94\u7528\u573a\u666f",id:"\u5e94\u7528\u573a\u666f-2",level:4},{value:"\u4e09\u79cd\u6280\u672f\u5bf9\u6bd4",id:"\u4e09\u79cd\u6280\u672f\u5bf9\u6bd4",level:3},{value:"Python\u68c0\u6d4bBLE\u4fe1\u53f7",id:"python\u68c0\u6d4bble\u4fe1\u53f7",level:2},{value:"\u6240\u9700\u5e93\u548c\u4f9d\u8d56",id:"\u6240\u9700\u5e93\u548c\u4f9d\u8d56",level:3},{value:"\u4ee3\u7801\u7ed3\u6784",id:"\u4ee3\u7801\u7ed3\u6784",level:3},{value:"\u521d\u59cb\u5316\u548c\u914d\u7f6e",id:"\u521d\u59cb\u5316\u548c\u914d\u7f6e",level:3},{value:"BLE\u8bbe\u5907\u626b\u63cf",id:"ble\u8bbe\u5907\u626b\u63cf",level:3},{value:"\u4fe1\u6807\u6570\u636e\u89e3\u6790",id:"\u4fe1\u6807\u6570\u636e\u89e3\u6790",level:3},{value:"\u6570\u636e\u5904\u7406\u548c\u53ef\u89c6\u5316",id:"\u6570\u636e\u5904\u7406\u548c\u53ef\u89c6\u5316",level:3},{value:"\u5b8c\u6574\u793a\u4f8b",id:"\u5b8c\u6574\u793a\u4f8b",level:3},{value:"\u4fe1\u53f7\u5f3a\u5ea6\u53ef\u89c6\u5316",id:"\u4fe1\u53f7\u5f3a\u5ea6\u53ef\u89c6\u5316",level:3},{value:"\u603b\u7ed3",id:"\u603b\u7ed3",level:2},{value:"\u53c2\u8003\u8d44\u6599",id:"\u53c2\u8003\u8d44\u6599",level:2}];function o(n){const e={a:"a",code:"code",h2:"h2",h3:"h3",h4:"h4",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,d.R)(),...n.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(e.p,{children:"\u672c\u6587\u4ecb\u7ecd\u4e86\u5982\u4f55\u4f7f\u7528Python\u5728MacOS\u4e0a\u68c0\u6d4bBLE\u4fe1\u53f7\u5e76\u53ef\u89c6\u5316\u5c55\u793a\u4fe1\u53f7\u5f3a\u5ea6\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"\u5e38\u89c1\u7684\u5b9a\u4f4d\u65b9\u6cd5",children:"\u5e38\u89c1\u7684\u5b9a\u4f4d\u65b9\u6cd5"}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u8bbe\u5907\u5b9a\u4f4d\u7684\u9886\u57df\u5185\uff0c\u6709\u5927\u69823\u79cd\u5b9a\u4f4d\u6280\u672f\uff0c\u5176\u5206\u522b\u4e3a\uff1aUWB\uff08\u8d85\u5bbd\u5e26\uff09\u3001BLE\uff08\u4f4e\u529f\u8017\u84dd\u7259\uff09\u548cWiFi\u3002\u8fd9\u4e09\u79cd\u6280\u672f\u5404\u6709\u4f18\u7f3a\u70b9\uff0c\u9002\u7528\u4e8e\u4e0d\u540c\u7684\u573a\u666f\u3002"}),"\n",(0,l.jsx)(e.h3,{id:"uwbultra-wideband\u8d85\u5bbd\u5e26\u5b9a\u4f4d",children:"UWB\uff08Ultra-Wideband\uff0c\u8d85\u5bbd\u5e26\uff09\u5b9a\u4f4d"}),"\n",(0,l.jsx)(e.p,{children:"UWB\u662f\u4e00\u79cd\u4f7f\u7528\u6781\u77ed\u8109\u51b2\u5728\u5bbd\u9891\u5e26\u4e0a\u4f20\u8f93\u6570\u636e\u7684\u65e0\u7ebf\u901a\u4fe1\u6280\u672f\u3002\u5728\u5b9a\u4f4d\u9886\u57df\uff0cUWB\u5177\u6709\u4ee5\u4e0b\u7279\u70b9\uff1a"}),"\n",(0,l.jsx)(e.h4,{id:"\u5de5\u4f5c\u539f\u7406",children:"\u5de5\u4f5c\u539f\u7406"}),"\n",(0,l.jsx)(e.p,{children:"UWB\u5b9a\u4f4d\u4e3b\u8981\u57fa\u4e8eTOF\uff08Time of Flight\uff0c\u98de\u884c\u65f6\u95f4\uff09\u6216TDOA\uff08Time Difference of Arrival\uff0c\u5230\u8fbe\u65f6\u95f4\u5dee\uff09\u539f\u7406\u3002\u8bbe\u5907\u901a\u8fc7\u6d4b\u91cf\u65e0\u7ebf\u7535\u4fe1\u53f7\u4ece\u53d1\u5c04\u5230\u63a5\u6536\u7684\u65f6\u95f4\u6765\u8ba1\u7b97\u8ddd\u79bb\uff0c\u8fdb\u800c\u786e\u5b9a\u4f4d\u7f6e\u3002"}),"\n",(0,l.jsx)(e.h4,{id:"\u4f18\u52bf",children:"\u4f18\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u9ad8\u7cbe\u5ea6"}),"\uff1aUWB\u53ef\u4ee5\u63d0\u4f9b\u5398\u7c73\u7ea7\u7684\u5b9a\u4f4d\u7cbe\u5ea6\uff08\u901a\u5e38\u572810-30\u5398\u7c73\u8303\u56f4\u5185\uff09"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6297\u591a\u5f84\u5e72\u6270"}),"\uff1a\u7531\u4e8e\u4f7f\u7528\u6781\u77ed\u7684\u8109\u51b2\uff0cUWB\u5bf9\u591a\u5f84\u5e72\u6270\u6709\u5f88\u5f3a\u7684\u62b5\u6297\u529b"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u7a7f\u5899\u80fd\u529b\u5f3a"}),"\uff1aUWB\u4fe1\u53f7\u53ef\u4ee5\u7a7f\u900f\u5899\u58c1\u548c\u5176\u4ed6\u969c\u788d\u7269"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u4f4e\u529f\u8017"}),"\uff1a\u76f8\u5bf9\u4e8e\u5176\u4ed6\u9ad8\u7cbe\u5ea6\u5b9a\u4f4d\u6280\u672f\uff0cUWB\u7684\u529f\u8017\u8f83\u4f4e"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u52a3\u52bf",children:"\u52a3\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6210\u672c\u9ad8"}),"\uff1aUWB\u8bbe\u5907\u548c\u57fa\u7840\u8bbe\u65bd\u7684\u6210\u672c\u76f8\u5bf9\u8f83\u9ad8"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u8986\u76d6\u8303\u56f4\u6709\u9650"}),"\uff1a\u901a\u5e38\u9700\u8981\u591a\u4e2a\u951a\u70b9\uff08\u57fa\u7ad9\uff09\u6765\u8986\u76d6\u8f83\u5927\u533a\u57df"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6807\u51c6\u5316\u7a0b\u5ea6\u8f83\u4f4e"}),"\uff1a\u867d\u7136\u6709IEEE 802.15.4z\u6807\u51c6\uff0c\u4f46\u5e02\u573a\u4e0a\u7684\u5b9e\u73b0\u591a\u6837\u5316"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u5e94\u7528\u573a\u666f",children:"\u5e94\u7528\u573a\u666f"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsx)(e.li,{children:"\u9ad8\u7cbe\u5ea6\u5ba4\u5185\u5b9a\u4f4d"}),"\n",(0,l.jsx)(e.li,{children:"\u8d44\u4ea7\u8ffd\u8e2a"}),"\n",(0,l.jsx)(e.li,{children:"\u667a\u80fd\u5bb6\u5c45"}),"\n",(0,l.jsx)(e.li,{children:"\u5de5\u4e1a\u81ea\u52a8\u5316"}),"\n",(0,l.jsx)(e.li,{children:"\u8f66\u8f86\u9632\u76d7\u7cfb\u7edf\uff08\u5982Apple AirTag\u7b49\uff09"}),"\n"]}),"\n",(0,l.jsx)(e.h3,{id:"blebluetooth-low-energy\u4f4e\u529f\u8017\u84dd\u7259\u5b9a\u4f4d",children:"BLE\uff08Bluetooth Low Energy\uff0c\u4f4e\u529f\u8017\u84dd\u7259\uff09\u5b9a\u4f4d"}),"\n",(0,l.jsx)(e.p,{children:"BLE\u662f\u84dd\u7259\u6280\u672f\u7684\u4e00\u4e2a\u5b50\u96c6\uff0c\u4e13\u4e3a\u4f4e\u529f\u8017\u5e94\u7528\u8bbe\u8ba1\u3002\u5728\u5b9a\u4f4d\u9886\u57df\uff0cBLE\u4e3b\u8981\u901a\u8fc7\u4fe1\u6807\uff08Beacon\uff09\u6280\u672f\u5b9e\u73b0\u3002"}),"\n",(0,l.jsx)(e.h4,{id:"\u5de5\u4f5c\u539f\u7406-1",children:"\u5de5\u4f5c\u539f\u7406"}),"\n",(0,l.jsx)(e.p,{children:"BLE\u5b9a\u4f4d\u4e3b\u8981\u57fa\u4e8eRSSI\uff08Received Signal Strength Indication\uff0c\u63a5\u6536\u4fe1\u53f7\u5f3a\u5ea6\u6307\u793a\uff09\u3002\u901a\u8fc7\u6d4b\u91cf\u63a5\u6536\u5230\u7684\u4fe1\u53f7\u5f3a\u5ea6\uff0c\u5e76\u7ed3\u5408\u8def\u5f84\u635f\u8017\u6a21\u578b\uff0c\u53ef\u4ee5\u4f30\u7b97\u8bbe\u5907\u4e0e\u4fe1\u6807\u4e4b\u95f4\u7684\u8ddd\u79bb\u3002\u5e38\u89c1\u7684\u534f\u8bae\u5305\u62eciBeacon\uff08\u82f9\u679c\uff09\u548cEddystone\uff08\u8c37\u6b4c\uff09\u3002"}),"\n",(0,l.jsx)(e.h4,{id:"\u4f18\u52bf-1",children:"\u4f18\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u4f4e\u529f\u8017"}),"\uff1aBLE\u8bbe\u5907\u53ef\u4ee5\u4f7f\u7528\u7ebd\u6263\u7535\u6c60\u8fd0\u884c\u6570\u6708\u751a\u81f3\u6570\u5e74"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6210\u672c\u4f4e"}),"\uff1aBLE\u82af\u7247\u548c\u4fe1\u6807\u4ef7\u683c\u4fbf\u5b9c\uff0c\u90e8\u7f72\u6210\u672c\u4f4e"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u517c\u5bb9\u6027\u597d"}),"\uff1a\u51e0\u4e4e\u6240\u6709\u73b0\u4ee3\u667a\u80fd\u624b\u673a\u90fd\u652f\u6301BLE"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u90e8\u7f72\u7b80\u5355"}),"\uff1a\u65e0\u9700\u590d\u6742\u7684\u57fa\u7840\u8bbe\u65bd"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u52a3\u52bf-1",children:"\u52a3\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u7cbe\u5ea6\u6709\u9650"}),"\uff1a\u5178\u578b\u7cbe\u5ea6\u57283-5\u7c73\uff0c\u53d7\u73af\u5883\u5f71\u54cd\u5927"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6613\u53d7\u5e72\u6270"}),"\uff1a\u4fe1\u53f7\u5bb9\u6613\u53d7\u5230\u4eba\u4f53\u3001\u5899\u58c1\u7b49\u969c\u788d\u7269\u7684\u5f71\u54cd"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u8ddd\u79bb\u6709\u9650"}),"\uff1a\u6709\u6548\u8303\u56f4\u901a\u5e38\u572850\u7c73\u4ee5\u5185"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u5e94\u7528\u573a\u666f-1",children:"\u5e94\u7528\u573a\u666f"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsx)(e.li,{children:"\u5546\u573a\u5bfc\u822a"}),"\n",(0,l.jsx)(e.li,{children:"\u5c55\u89c8\u4f1a\u4fe1\u606f\u63a8\u9001"}),"\n",(0,l.jsx)(e.li,{children:"\u8d44\u4ea7\u8ffd\u8e2a"}),"\n",(0,l.jsx)(e.li,{children:"\u8003\u52e4\u7cfb\u7edf"}),"\n",(0,l.jsx)(e.li,{children:"\u667a\u80fd\u5bb6\u5c45\u81ea\u52a8\u5316"}),"\n"]}),"\n",(0,l.jsx)(e.h3,{id:"wifi\u5b9a\u4f4d",children:"WiFi\u5b9a\u4f4d"}),"\n",(0,l.jsx)(e.p,{children:"WiFi\u5b9a\u4f4d\u5229\u7528\u73b0\u6709\u7684WiFi\u57fa\u7840\u8bbe\u65bd\u8fdb\u884c\u5ba4\u5185\u5b9a\u4f4d\uff0c\u662f\u6700\u5e7f\u6cdb\u90e8\u7f72\u7684\u5ba4\u5185\u5b9a\u4f4d\u6280\u672f\u4e4b\u4e00\u3002"}),"\n",(0,l.jsx)(e.h4,{id:"\u5de5\u4f5c\u539f\u7406-2",children:"\u5de5\u4f5c\u539f\u7406"}),"\n",(0,l.jsx)(e.p,{children:"WiFi\u5b9a\u4f4d\u4e3b\u8981\u6709\u4e24\u79cd\u65b9\u5f0f\uff1a"}),"\n",(0,l.jsxs)(e.ol,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u57fa\u4e8eRSSI\u7684\u4e09\u8fb9\u6d4b\u91cf"}),"\uff1a\u901a\u8fc7\u6d4b\u91cf\u8bbe\u5907\u4e0e\u591a\u4e2aWiFi\u63a5\u5165\u70b9\u4e4b\u95f4\u7684\u4fe1\u53f7\u5f3a\u5ea6\uff0c\u4f30\u7b97\u8ddd\u79bb\u5e76\u786e\u5b9a\u4f4d\u7f6e"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6307\u7eb9\u5b9a\u4f4d"}),'\uff1a\u9884\u5148\u91c7\u96c6\u7a7a\u95f4\u4e2d\u5404\u70b9\u7684WiFi\u4fe1\u53f7\u7279\u5f81\uff0c\u5f62\u6210"\u6307\u7eb9\u6570\u636e\u5e93"\uff0c\u5b9a\u4f4d\u65f6\u5c06\u5b9e\u65f6\u91c7\u96c6\u7684\u4fe1\u53f7\u4e0e\u6570\u636e\u5e93\u5339\u914d']}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u4f18\u52bf-2",children:"\u4f18\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u57fa\u7840\u8bbe\u65bd\u5e7f\u6cdb"}),"\uff1a\u5229\u7528\u73b0\u6709WiFi\u7f51\u7edc\uff0c\u65e0\u9700\u989d\u5916\u786c\u4ef6"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u8986\u76d6\u8303\u56f4\u5927"}),"\uff1a\u5355\u4e2a\u63a5\u5165\u70b9\u53ef\u8986\u76d6\u6570\u5341\u7c73\u8303\u56f4"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6210\u672c\u4f4e"}),"\uff1a\u5982\u679c\u5df2\u6709WiFi\u7f51\u7edc\uff0c\u51e0\u4e4e\u65e0\u989d\u5916\u6210\u672c"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u517c\u5bb9\u6027\u597d"}),"\uff1a\u51e0\u4e4e\u6240\u6709\u79fb\u52a8\u8bbe\u5907\u90fd\u652f\u6301WiFi"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u52a3\u52bf-2",children:"\u52a3\u52bf"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u7cbe\u5ea6\u4e00\u822c"}),"\uff1a\u5178\u578b\u7cbe\u5ea6\u57283-15\u7c73\uff0c\u53d6\u51b3\u4e8e\u73af\u5883\u548c\u63a5\u5165\u70b9\u5bc6\u5ea6"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6613\u53d7\u5e72\u6270"}),"\uff1a\u4fe1\u53f7\u53d7\u73af\u5883\u53d8\u5316\u5f71\u54cd\u5927"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u529f\u8017\u8f83\u9ad8"}),"\uff1a\u76f8\u6bd4BLE\u548cUWB\uff0cWiFi\u7684\u529f\u8017\u8f83\u9ad8"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u521d\u59cb\u5316\u590d\u6742"}),"\uff1a\u6307\u7eb9\u5b9a\u4f4d\u9700\u8981\u524d\u671f\u5927\u91cf\u91c7\u96c6\u5de5\u4f5c"]}),"\n"]}),"\n",(0,l.jsx)(e.h4,{id:"\u5e94\u7528\u573a\u666f-2",children:"\u5e94\u7528\u573a\u666f"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsx)(e.li,{children:"\u5927\u578b\u5efa\u7b51\u7269\u5185\u5bfc\u822a"}),"\n",(0,l.jsx)(e.li,{children:"\u5546\u573a\u5ba2\u6d41\u5206\u6790"}),"\n",(0,l.jsx)(e.li,{children:"\u516c\u5171\u573a\u6240\u4f4d\u7f6e\u670d\u52a1"}),"\n",(0,l.jsx)(e.li,{children:"\u8d44\u4ea7\u7ba1\u7406"}),"\n",(0,l.jsx)(e.li,{children:"\u667a\u80fd\u529e\u516c"}),"\n"]}),"\n",(0,l.jsx)(e.h3,{id:"\u4e09\u79cd\u6280\u672f\u5bf9\u6bd4",children:"\u4e09\u79cd\u6280\u672f\u5bf9\u6bd4"}),"\n",(0,l.jsxs)(e.table,{children:[(0,l.jsx)(e.thead,{children:(0,l.jsxs)(e.tr,{children:[(0,l.jsx)(e.th,{children:"\u6280\u672f"}),(0,l.jsx)(e.th,{children:"\u7cbe\u5ea6"}),(0,l.jsx)(e.th,{children:"\u529f\u8017"}),(0,l.jsx)(e.th,{children:"\u6210\u672c"}),(0,l.jsx)(e.th,{children:"\u8986\u76d6\u8303\u56f4"}),(0,l.jsx)(e.th,{children:"\u6297\u5e72\u6270\u80fd\u529b"})]})}),(0,l.jsxs)(e.tbody,{children:[(0,l.jsxs)(e.tr,{children:[(0,l.jsx)(e.td,{children:"UWB"}),(0,l.jsx)(e.td,{children:"10-30\u5398\u7c73"}),(0,l.jsx)(e.td,{children:"\u4e2d\u7b49"}),(0,l.jsx)(e.td,{children:"\u9ad8"}),(0,l.jsx)(e.td,{children:"\u5c0f\uff08~50\u7c73\uff09"}),(0,l.jsx)(e.td,{children:"\u5f3a"})]}),(0,l.jsxs)(e.tr,{children:[(0,l.jsx)(e.td,{children:"BLE"}),(0,l.jsx)(e.td,{children:"3-5\u7c73"}),(0,l.jsx)(e.td,{children:"\u4f4e"}),(0,l.jsx)(e.td,{children:"\u4f4e"}),(0,l.jsx)(e.td,{children:"\u4e2d\uff08~50\u7c73\uff09"}),(0,l.jsx)(e.td,{children:"\u5f31"})]}),(0,l.jsxs)(e.tr,{children:[(0,l.jsx)(e.td,{children:"WiFi"}),(0,l.jsx)(e.td,{children:"3-15\u7c73"}),(0,l.jsx)(e.td,{children:"\u9ad8"}),(0,l.jsx)(e.td,{children:"\u4f4e\uff08\u5229\u7528\u73b0\u6709\u7f51\u7edc\uff09"}),(0,l.jsx)(e.td,{children:"\u5927\uff08~100\u7c73\uff09"}),(0,l.jsx)(e.td,{children:"\u4e2d"})]})]})]}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u5b9e\u9645\u5e94\u7528\u4e2d\uff0c\u8fd9\u4e09\u79cd\u6280\u672f\u5f80\u5f80\u4f1a\u7ed3\u5408\u4f7f\u7528\uff0c\u4ee5\u5f25\u8865\u5404\u81ea\u7684\u4e0d\u8db3\u3002\u4f8b\u5982\uff0c\u53ef\u4ee5\u4f7f\u7528WiFi\u8fdb\u884c\u7c97\u7565\u5b9a\u4f4d\uff0c\u7136\u540e\u4f7f\u7528BLE\u8fdb\u884c\u533a\u57df\u786e\u8ba4\uff0c\u6700\u540e\u5728\u9700\u8981\u9ad8\u7cbe\u5ea6\u7684\u573a\u666f\u4e0b\u4f7f\u7528UWB\u8fdb\u884c\u7cbe\u786e\u5b9a\u4f4d\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u63a5\u4e0b\u6765\uff0c\u6211\u4eec\u5c06\u91cd\u70b9\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528Python\u68c0\u6d4bBLE\u4fe1\u53f7\u5e76\u53ef\u89c6\u5316\u5c55\u793a\u4fe1\u53f7\u5f3a\u5ea6\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"python\u68c0\u6d4bble\u4fe1\u53f7",children:"Python\u68c0\u6d4bBLE\u4fe1\u53f7"}),"\n",(0,l.jsxs)(e.p,{children:["\u5728\u672c\u8282\u4e2d\uff0c\u6211\u4eec\u5c06\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528Python\u6765\u68c0\u6d4b\u548c\u5206\u6790BLE\u4fe1\u53f7\u3002\u6211\u4eec\u5c06\u57fa\u4e8e",(0,l.jsx)(e.a,{href:"https://github.com/yishi-projects/ble-beacon",children:"yishi-projects/ble-beacon"}),"\u9879\u76ee\u4e2d\u7684\u4ee3\u7801\u6765\u5b9e\u73b0\u8fd9\u4e00\u529f\u80fd\u3002"]}),"\n",(0,l.jsx)(e.h3,{id:"\u6240\u9700\u5e93\u548c\u4f9d\u8d56",children:"\u6240\u9700\u5e93\u548c\u4f9d\u8d56"}),"\n",(0,l.jsx)(e.p,{children:"\u9996\u5148\uff0c\u6211\u4eec\u9700\u8981\u5b89\u88c5\u4ee5\u4e0bPython\u5e93\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-bash",children:"pip install bleak kafka-python\n"})}),"\n",(0,l.jsx)(e.p,{children:"\u4e3b\u8981\u4f9d\u8d56\u5305\u62ec\uff1a"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"bleak"}),"\uff1a\u8de8\u5e73\u53f0\u7684BLE\u5ba2\u6237\u7aef\u5e93\uff0c\u652f\u6301Windows\u3001macOS\u548cLinux"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"kafka-python"}),"\uff1a\u7528\u4e8e\u5c06\u6570\u636e\u53d1\u9001\u5230Kafka\uff08\u53ef\u9009\uff0c\u7528\u4e8e\u6570\u636e\u6d41\u5904\u7406\uff09"]}),"\n"]}),"\n",(0,l.jsx)(e.h3,{id:"\u4ee3\u7801\u7ed3\u6784",children:"\u4ee3\u7801\u7ed3\u6784"}),"\n",(0,l.jsx)(e.p,{children:"\u6211\u4eec\u7684BLE\u68c0\u6d4b\u7a0b\u5e8f\u4e3b\u8981\u5305\u542b\u4ee5\u4e0b\u51e0\u4e2a\u90e8\u5206\uff1a"}),"\n",(0,l.jsxs)(e.ol,{children:["\n",(0,l.jsx)(e.li,{children:"\u521d\u59cb\u5316\u548c\u914d\u7f6e"}),"\n",(0,l.jsx)(e.li,{children:"BLE\u8bbe\u5907\u626b\u63cf"}),"\n",(0,l.jsx)(e.li,{children:"\u4fe1\u6807\u6570\u636e\u89e3\u6790\uff08iBeacon\u3001Eddystone\u7b49\uff09"}),"\n",(0,l.jsx)(e.li,{children:"\u6570\u636e\u5904\u7406\u548c\u53ef\u89c6\u5316"}),"\n"]}),"\n",(0,l.jsx)(e.h3,{id:"\u521d\u59cb\u5316\u548c\u914d\u7f6e",children:"\u521d\u59cb\u5316\u548c\u914d\u7f6e"}),"\n",(0,l.jsx)(e.p,{children:"\u9996\u5148\uff0c\u6211\u4eec\u9700\u8981\u5bfc\u5165\u5fc5\u8981\u7684\u5e93\u5e76\u8bbe\u7f6e\u57fa\u672c\u914d\u7f6e\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:"import asyncio\nfrom bleak import BleakScanner\nimport uuid\nimport time\nimport datetime\nimport os\nimport configparser\n\n# \u5168\u5c40\u53d8\u91cf\u63a7\u5236\u626b\u63cf\u72b6\u6001\n_scanning_active = False\n\n# \u52a0\u8f7d\u914d\u7f6e\u6587\u4ef6\ndef load_config():\n    \"\"\"\u4ece~/.ble/config.conf\u52a0\u8f7d\u914d\u7f6e\uff0c\u5982\u679c\u4e0d\u5b58\u5728\u5219\u521b\u5efa\u9ed8\u8ba4\u914d\u7f6e\"\"\"\n    config = configparser.ConfigParser()\n    \n    # \u9ed8\u8ba4\u914d\u7f6e\n    config['kafka'] = {\n        'broker': 'localhost:9092',\n        'topic': 'ble_beacons'\n    }\n    \n    # \u521b\u5efa\u914d\u7f6e\u76ee\u5f55\uff08\u5982\u679c\u4e0d\u5b58\u5728\uff09\n    config_dir = os.path.expanduser(\"~/.ble\")\n    os.makedirs(config_dir, exist_ok=True)\n    \n    config_file = os.path.join(config_dir, \"config.conf\")\n    \n    # \u5982\u679c\u914d\u7f6e\u6587\u4ef6\u5b58\u5728\uff0c\u8bfb\u53d6\u5b83\n    if os.path.exists(config_file):\n        config.read(config_file)\n    else:\n        # \u521b\u5efa\u9ed8\u8ba4\u914d\u7f6e\u6587\u4ef6\n        with open(config_file, 'w') as f:\n            config.write(f)\n    \n    return config\n"})}),"\n",(0,l.jsx)(e.h3,{id:"ble\u8bbe\u5907\u626b\u63cf",children:"BLE\u8bbe\u5907\u626b\u63cf"}),"\n",(0,l.jsxs)(e.p,{children:["BLE\u8bbe\u5907\u626b\u63cf\u662f\u6574\u4e2a\u7a0b\u5e8f\u7684\u6838\u5fc3\u90e8\u5206\u3002\u6211\u4eec\u4f7f\u7528",(0,l.jsx)(e.code,{children:"bleak"}),"\u5e93\u7684",(0,l.jsx)(e.code,{children:"BleakScanner"}),"\u6765\u5f02\u6b65\u626b\u63cf\u5468\u56f4\u7684BLE\u8bbe\u5907\uff1a"]}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:'async def scan_ble_devices():\n    """\u626b\u63cfBLE\u8bbe\u5907\u5e76\u5904\u7406\u4fe1\u6807\u6570\u636e"""\n    # \u83b7\u53d6\u4e3b\u673aID\uff08\u7528\u4e8e\u6570\u636e\u6807\u8bc6\uff09\n    host_id = get_host_id()\n    \n    # \u8ba1\u6570\u5668\uff08\u7528\u4e8e\u65e5\u5fd7\uff09\n    scan_count = 0\n    \n    # \u8bbe\u7f6e\u626b\u63cf\u72b6\u6001\n    global _scanning_active\n    _scanning_active = True\n    \n    try:\n        # \u6301\u7eed\u626b\u63cf\u5faa\u73af\n        while _scanning_active:\n            scan_count += 1\n            \n            # \u626b\u63cf\u8bbe\u5907\uff08\u8d85\u65f61\u79d2\uff09\n            devices = await BleakScanner.discover(timeout=1.0)\n            \n            # \u68c0\u67e5\u662f\u5426\u5e94\u8be5\u505c\u6b62\u626b\u63cf\n            if not _scanning_active:\n                break\n            \n            # \u5904\u7406\u6bcf\u4e2a\u8bbe\u5907\n            beacons_found = 0\n            for device in devices:\n                # \u63d0\u53d6\u5236\u9020\u5546\u6570\u636e\n                if device.metadata.get(\'manufacturer_data\'):\n                    for company_code, data in device.metadata[\'manufacturer_data\'].items():\n                        # \u68c0\u6d4b\u4e0d\u540c\u7c7b\u578b\u7684\u4fe1\u6807\n                        process_beacon_data(company_code, data, device)\n            \n            # \u7b49\u5f85\u4e0b\u4e00\u6b21\u626b\u63cf\n            await asyncio.sleep(0)\n    finally:\n        # \u6e05\u7406\u8d44\u6e90\n        pass\n'})}),"\n",(0,l.jsx)(e.h3,{id:"\u4fe1\u6807\u6570\u636e\u89e3\u6790",children:"\u4fe1\u6807\u6570\u636e\u89e3\u6790"}),"\n",(0,l.jsx)(e.p,{children:"BLE\u4fe1\u6807\u6709\u591a\u79cd\u7c7b\u578b\uff0c\u6700\u5e38\u89c1\u7684\u662fiBeacon\uff08\u82f9\u679c\uff09\u548cEddystone\uff08\u8c37\u6b4c\uff09\u3002\u6211\u4eec\u9700\u8981\u6839\u636e\u4e0d\u540c\u7684\u534f\u8bae\u683c\u5f0f\u89e3\u6790\u6570\u636e\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:"def process_beacon_data(company_code, data, device):\n    \"\"\"\u6839\u636e\u4e0d\u540c\u7684\u4fe1\u6807\u7c7b\u578b\u89e3\u6790\u6570\u636e\"\"\"\n    # \u68c0\u67e5iBeacon\uff08\u82f9\u679c\u516c\u53f8\u4ee3\u7801\u662f0x004C\uff09\n    if company_code == 0x004C and len(data) >= 23:\n        try:\n            # \u68c0\u67e5iBeacon\u6807\u8bc6\u7b26\uff080x02, 0x15\uff09\n            if data[0] == 0x02 and data[1] == 0x15:\n                # \u89e3\u6790iBeacon\u6570\u636e\n                uuid_bytes = data[2:18]\n                uuid_str = str(uuid.UUID(bytes=bytes(uuid_bytes)))\n                major = int.from_bytes(data[18:20], byteorder='big')\n                minor = int.from_bytes(data[20:22], byteorder='big')\n                tx_power = data[22] - 256 if data[22] > 127 else data[22]\n                \n                beacon_data = {\n                    'uuid': uuid_str,\n                    'major': major,\n                    'minor': minor,\n                    'tx_power': tx_power,\n                    'rssi': device.rssi,\n                    'address': device.address,\n                    'name': device.name or 'Unknown'\n                }\n                \n                # \u5904\u7406iBeacon\u6570\u636e\n                process_beacon('iBeacon', beacon_data)\n        except Exception as e:\n            print(f\"\u5904\u7406iBeacon\u6570\u636e\u65f6\u51fa\u9519: {e}\")\n    \n    # \u68c0\u67e5Eddystone\u4fe1\u6807\uff08\u8c37\u6b4c\u516c\u53f8\u4ee3\u7801\u662f0x00AA\uff09\n    elif company_code == 0x00AA and len(data) >= 20:\n        try:\n            # \u68c0\u67e5Eddystone\u6807\u8bc6\u7b26\n            if data[0] == 0xAA and data[1] == 0xFE:\n                frame_type = data[2]\n                \n                # Eddystone-UID\n                if frame_type == 0x00:\n                    namespace = bytes(data[3:13]).hex()\n                    instance = bytes(data[13:19]).hex()\n                    \n                    beacon_data = {\n                        'namespace': namespace,\n                        'instance': instance,\n                        'rssi': device.rssi,\n                        'address': device.address,\n                        'name': device.name or 'Unknown'\n                    }\n                    \n                    # \u5904\u7406Eddystone-UID\u6570\u636e\n                    process_beacon('Eddystone-UID', beacon_data)\n                \n                # Eddystone-URL\n                elif frame_type == 0x10:\n                    url_scheme = ['http://www.', 'https://www.', 'http://', 'https://'][data[3]]\n                    url_data = bytes(data[4:]).decode('ascii')\n                    url = url_scheme + url_data\n                    \n                    beacon_data = {\n                        'url': url,\n                        'rssi': device.rssi,\n                        'address': device.address,\n                        'name': device.name or 'Unknown'\n                    }\n                    \n                    # \u5904\u7406Eddystone-URL\u6570\u636e\n                    process_beacon('Eddystone-URL', beacon_data)\n        except Exception as e:\n            print(f\"\u5904\u7406Eddystone\u6570\u636e\u65f6\u51fa\u9519: {e}\")\n"})}),"\n",(0,l.jsx)(e.h3,{id:"\u6570\u636e\u5904\u7406\u548c\u53ef\u89c6\u5316",children:"\u6570\u636e\u5904\u7406\u548c\u53ef\u89c6\u5316"}),"\n",(0,l.jsx)(e.p,{children:"\u6536\u96c6\u5230\u7684BLE\u4fe1\u53f7\u6570\u636e\u53ef\u4ee5\u901a\u8fc7\u591a\u79cd\u65b9\u5f0f\u8fdb\u884c\u5904\u7406\u548c\u53ef\u89c6\u5316\uff1a"}),"\n",(0,l.jsxs)(e.ol,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u5b9e\u65f6\u663e\u793a"}),"\uff1a\u4f7f\u7528GUI\u5e93\uff08\u5982Tkinter\u3001PyQt\u7b49\uff09\u5b9e\u65f6\u663e\u793a\u68c0\u6d4b\u5230\u7684\u8bbe\u5907\u548c\u4fe1\u53f7\u5f3a\u5ea6"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6570\u636e\u5b58\u50a8"}),"\uff1a\u5c06\u6570\u636e\u4fdd\u5b58\u5230\u672c\u5730\u6587\u4ef6\u6216\u6570\u636e\u5e93\u4e2d"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u6570\u636e\u6d41\u5904\u7406"}),"\uff1a\u4f7f\u7528Kafka\u7b49\u6d88\u606f\u961f\u5217\u8fdb\u884c\u5b9e\u65f6\u6570\u636e\u6d41\u5904\u7406"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.strong,{children:"\u4fe1\u53f7\u5f3a\u5ea6\u53ef\u89c6\u5316"}),"\uff1a\u4f7f\u7528matplotlib\u7b49\u5e93\u7ed8\u5236\u4fe1\u53f7\u5f3a\u5ea6\u70ed\u56fe\u6216\u65f6\u95f4\u5e8f\u5217\u56fe"]}),"\n"]}),"\n",(0,l.jsx)(e.p,{children:"\u4ee5\u4e0b\u662f\u4e00\u4e2a\u7b80\u5355\u7684\u6570\u636e\u5904\u7406\u51fd\u6570\u793a\u4f8b\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:"def process_beacon(beacon_type, beacon_data):\n    \"\"\"\u5904\u7406\u4fe1\u6807\u6570\u636e\"\"\"\n    timestamp = datetime.datetime.now().isoformat()\n    \n    # \u6dfb\u52a0\u901a\u7528\u5b57\u6bb5\n    message = {\n        'type': beacon_type,\n        'timestamp': timestamp,\n        'rssi': beacon_data.get('rssi', 0),\n        'address': beacon_data.get('address', 'unknown')\n    }\n    \n    # \u6dfb\u52a0\u7279\u5b9a\u7c7b\u578b\u7684\u5b57\u6bb5\n    message.update(beacon_data)\n    \n    # \u8fd9\u91cc\u53ef\u4ee5\u6dfb\u52a0\u6570\u636e\u5904\u7406\u903b\u8f91\n    # \u4f8b\u5982\uff1a\u4fdd\u5b58\u5230\u6587\u4ef6\u3001\u53d1\u9001\u5230\u670d\u52a1\u5668\u3001\u66f4\u65b0GUI\u7b49\n    \n    return message\n"})}),"\n",(0,l.jsx)(e.h3,{id:"\u5b8c\u6574\u793a\u4f8b",children:"\u5b8c\u6574\u793a\u4f8b"}),"\n",(0,l.jsx)(e.p,{children:"\u4e0b\u9762\u662f\u4e00\u4e2a\u7b80\u5355\u4f46\u5b8c\u6574\u7684BLE\u626b\u63cf\u5668\u793a\u4f8b\uff0c\u5b83\u4f1a\u626b\u63cf\u5468\u56f4\u7684BLE\u8bbe\u5907\u5e76\u6253\u5370\u51fa\u68c0\u6d4b\u5230\u7684\u4fe1\u6807\u4fe1\u606f\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:"import asyncio\nfrom bleak import BleakScanner\nimport uuid\nimport datetime\n\nasync def main():\n    print(\"\u5f00\u59cb\u626b\u63cfBLE\u8bbe\u5907...\")\n    \n    # \u626b\u63cf\u8bbe\u5907\n    devices = await BleakScanner.discover(timeout=5.0)\n    \n    print(f\"\u53d1\u73b0 {len(devices)} \u4e2a\u8bbe\u5907\")\n    \n    # \u5904\u7406\u6bcf\u4e2a\u8bbe\u5907\n    for device in devices:\n        print(f\"\u8bbe\u5907: {device.address} ({device.name or 'Unknown'}), RSSI: {device.rssi}\")\n        \n        # \u63d0\u53d6\u5236\u9020\u5546\u6570\u636e\n        if device.metadata.get('manufacturer_data'):\n            for company_code, data in device.metadata['manufacturer_data'].items():\n                # \u68c0\u67e5iBeacon\n                if company_code == 0x004C and len(data) >= 23:\n                    if data[0] == 0x02 and data[1] == 0x15:\n                        # \u89e3\u6790iBeacon\u6570\u636e\n                        uuid_bytes = data[2:18]\n                        uuid_str = str(uuid.UUID(bytes=bytes(uuid_bytes)))\n                        major = int.from_bytes(data[18:20], byteorder='big')\n                        minor = int.from_bytes(data[20:22], byteorder='big')\n                        \n                        print(f\"  iBeacon: UUID={uuid_str}, Major={major}, Minor={minor}\")\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n"})}),"\n",(0,l.jsx)(e.h3,{id:"\u4fe1\u53f7\u5f3a\u5ea6\u53ef\u89c6\u5316",children:"\u4fe1\u53f7\u5f3a\u5ea6\u53ef\u89c6\u5316"}),"\n",(0,l.jsx)(e.p,{children:"\u4e3a\u4e86\u66f4\u76f4\u89c2\u5730\u5c55\u793aBLE\u4fe1\u53f7\u5f3a\u5ea6\uff0c\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528matplotlib\u5e93\u521b\u5efa\u53ef\u89c6\u5316\u56fe\u8868\u3002\u4ee5\u4e0b\u662f\u4e00\u4e2a\u7b80\u5355\u7684\u793a\u4f8b\uff0c\u5c55\u793a\u5982\u4f55\u7ed8\u5236\u4fe1\u53f7\u5f3a\u5ea6\u968f\u65f6\u95f4\u53d8\u5316\u7684\u56fe\u8868\uff1a"}),"\n",(0,l.jsx)(e.pre,{children:(0,l.jsx)(e.code,{className:"language-python",children:"import matplotlib.pyplot as plt\nimport numpy as np\nimport time\nimport asyncio\nfrom bleak import BleakScanner\n\nasync def monitor_device(address, duration=60):\n    \"\"\"\u76d1\u63a7\u7279\u5b9a\u8bbe\u5907\u7684\u4fe1\u53f7\u5f3a\u5ea6\"\"\"\n    timestamps = []\n    rssi_values = []\n    \n    start_time = time.time()\n    end_time = start_time + duration\n    \n    while time.time() < end_time:\n        # \u626b\u63cf\u8bbe\u5907\n        devices = await BleakScanner.discover(timeout=1.0)\n        \n        # \u67e5\u627e\u76ee\u6807\u8bbe\u5907\n        for device in devices:\n            if device.address == address:\n                # \u8bb0\u5f55\u65f6\u95f4\u548cRSSI\n                timestamps.append(time.time() - start_time)\n                rssi_values.append(device.rssi)\n                print(f\"\u65f6\u95f4: {timestamps[-1]:.1f}s, RSSI: {rssi_values[-1]} dBm\")\n                break\n        \n        # \u7b49\u5f85\u4e0b\u4e00\u6b21\u626b\u63cf\n        await asyncio.sleep(0.5)\n    \n    # \u7ed8\u5236\u56fe\u8868\n    plt.figure(figsize=(10, 6))\n    plt.plot(timestamps, rssi_values, 'b-')\n    plt.xlabel('\u65f6\u95f4 (\u79d2)')\n    plt.ylabel('\u4fe1\u53f7\u5f3a\u5ea6 (dBm)')\n    plt.title(f'\u8bbe\u5907 {address} \u7684BLE\u4fe1\u53f7\u5f3a\u5ea6')\n    plt.grid(True)\n    plt.savefig('ble_signal_strength.png')\n    plt.show()\n\n# \u4f7f\u7528\u793a\u4f8b\n# asyncio.run(monitor_device('XX:XX:XX:XX:XX:XX', 60))\n"})}),"\n",(0,l.jsx)(e.h2,{id:"\u603b\u7ed3",children:"\u603b\u7ed3"}),"\n",(0,l.jsx)(e.p,{children:"\u901a\u8fc7Python\u548cbleak\u5e93\uff0c\u6211\u4eec\u53ef\u4ee5\u8f7b\u677e\u5730\u68c0\u6d4b\u548c\u5206\u6790BLE\u4fe1\u53f7\u3002\u8fd9\u79cd\u65b9\u6cd5\u9002\u7528\u4e8e\u591a\u79cd\u5e94\u7528\u573a\u666f\uff0c\u5982\u5ba4\u5185\u5b9a\u4f4d\u3001\u8d44\u4ea7\u8ffd\u8e2a\u3001\u5b58\u5728\u68c0\u6d4b\u7b49\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u5b9e\u9645\u5e94\u7528\u4e2d\uff0c\u6211\u4eec\u53ef\u4ee5\u6839\u636e\u9700\u8981\u6269\u5c55\u4e0a\u8ff0\u4ee3\u7801\uff0c\u4f8b\u5982\uff1a"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsx)(e.li,{children:"\u6dfb\u52a0\u8ddd\u79bb\u4f30\u7b97\uff08\u57fa\u4e8eRSSI\u548c\u8def\u5f84\u635f\u8017\u6a21\u578b\uff09"}),"\n",(0,l.jsx)(e.li,{children:"\u5b9e\u73b0\u4e09\u8fb9\u6d4b\u91cf\u5b9a\u4f4d\u7b97\u6cd5"}),"\n",(0,l.jsx)(e.li,{children:"\u5f00\u53d1\u5b9e\u65f6\u76d1\u63a7\u4eea\u8868\u677f"}),"\n",(0,l.jsx)(e.li,{children:"\u96c6\u6210\u673a\u5668\u5b66\u4e60\u7b97\u6cd5\u8fdb\u884c\u6a21\u5f0f\u8bc6\u522b"}),"\n"]}),"\n",(0,l.jsx)(e.p,{children:"BLE\u4fe1\u6807\u6280\u672f\u7ed3\u5408Python\u7684\u7075\u6d3b\u6027\uff0c\u4e3a\u6211\u4eec\u63d0\u4f9b\u4e86\u4e00\u4e2a\u5f3a\u5927\u7684\u5de5\u5177\uff0c\u53ef\u4ee5\u7528\u4e8e\u6784\u5efa\u5404\u79cd\u667a\u80fd\u7a7a\u95f4\u5e94\u7528\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"\u53c2\u8003\u8d44\u6599",children:"\u53c2\u8003\u8d44\u6599"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.a,{href:"https://github.com/yishi-projects/ble-beacon",children:"yishi-projects/ble-beacon"})," - BLE\u4fe1\u6807\u68c0\u6d4b\u9879\u76ee"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.a,{href:"https://bleak.readthedocs.io/",children:"Bleak\u6587\u6863"})," - \u8de8\u5e73\u53f0BLE\u5ba2\u6237\u7aef\u5e93"]}),"\n",(0,l.jsxs)(e.li,{children:[(0,l.jsx)(e.a,{href:"https://www.bluetooth.com/",children:"\u84dd\u7259SIG"})," - \u84dd\u7259\u6280\u672f\u6807\u51c6"]}),"\n"]})]})}function h(n={}){const{wrapper:e}={...(0,d.R)(),...n.components};return e?(0,l.jsx)(e,{...n,children:(0,l.jsx)(o,{...n})}):o(n)}}}]);