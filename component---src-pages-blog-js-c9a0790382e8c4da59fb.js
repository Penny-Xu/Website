(window.webpackJsonp=window.webpackJsonp||[]).push([[3],{145:function(e,a,t){"use strict";t.r(a);var n=t(159),l=t(0),r=t.n(l),c=t(154),s=t(150),i=t(160),o=t.n(i);a.default=function(){var e=n.data;return r.a.createElement("div",null,r.a.createElement(c.a,null,r.a.createElement("h1",null," Blog "),r.a.createElement("ol",{className:o.a.posts},e.allMarkdownRemark.edges.map(function(e){return r.a.createElement("li",{className:o.a.post},r.a.createElement(s.a,{to:"/blog/"+e.node.fields.slug},r.a.createElement("h2",null," ",e.node.frontmatter.title),r.a.createElement("p",null," ",e.node.frontmatter.date," ")))}))))}},149:function(e,a,t){var n;e.exports=(n=t(153))&&n.default||n},150:function(e,a,t){"use strict";var n=t(0),l=t.n(n),r=t(4),c=t.n(r),s=t(33),i=t.n(s);t.d(a,"a",function(){return i.a});t(149),l.a.createContext({});c.a.object,c.a.string.isRequired,c.a.func,c.a.func},151:function(e){e.exports={data:{site:{siteMetadata:{author:"Penny Xu"}}}}},152:function(e){e.exports={data:{site:{siteMetadata:{title:"Penny Xu"}}}}},153:function(e,a,t){"use strict";t.r(a);t(34);var n=t(0),l=t.n(n),r=t(4),c=t.n(r),s=t(55),i=t(2),o=function(e){var a=e.location,t=i.default.getResourcesForPathnameSync(a.pathname);return t?l.a.createElement(s.a,Object.assign({location:a,pageResources:t},t.json)):null};o.propTypes={location:c.a.shape({pathname:c.a.string.isRequired}).isRequired},a.default=o},154:function(e,a,t){"use strict";var n=t(0),l=t.n(n),r=t(151),c=t(155),s=t.n(c),i=function(){var e=r.data;return l.a.createElement("footer",{className:s.a.footer},l.a.createElement("p",null," Created by ",e.site.siteMetadata.author,", © 2019 "))},o=t(152),u=t(150),m=t(156),d=t.n(m),f=function(){var e=o.data;return l.a.createElement("header",{className:d.a.header},l.a.createElement("h1",null,l.a.createElement(u.a,{className:d.a.title,to:"/"}," ",e.site.siteMetadata.title," ")),l.a.createElement("nav",null,l.a.createElement("ul",{className:d.a.navList},l.a.createElement("li",null,l.a.createElement(u.a,{className:d.a.navItem,activeClassName:d.a.activeNavItem,to:"/"}," Home ")),l.a.createElement("li",null,l.a.createElement(u.a,{className:d.a.navItem,activeClassName:d.a.activeNavItem,to:"/blog"}," Blog ")),l.a.createElement("li",null,l.a.createElement(u.a,{className:d.a.navItem,activeClassName:d.a.activeNavItem,to:"/about"}," About Me ")),l.a.createElement("li",null,l.a.createElement(u.a,{className:d.a.navItem,activeClassName:d.a.activeNavItem,to:"/contact"}," Contact ")))))},v=(t(157),t(158)),E=t.n(v);a.a=function(e){return l.a.createElement("div",{className:E.a.container},l.a.createElement("div",{className:E.a.content},l.a.createElement(f,null),e.children),l.a.createElement(i,null))}},159:function(e){e.exports={data:{allMarkdownRemark:{edges:[{node:{frontmatter:{title:"Best Classifier",date:"05-01-2019"},fields:{slug:"p1"}}},{node:{frontmatter:{title:"Matrix Multiplication",date:"06-01-2019"},fields:{slug:"p2"}}}]}}}}}]);
//# sourceMappingURL=component---src-pages-blog-js-c9a0790382e8c4da59fb.js.map